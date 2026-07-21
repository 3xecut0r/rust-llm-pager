"""Diagnose the 7B-scale throughput regression at the kernel level, not the model level.

Local GPU (GTX 1050 Ti) can't load a 7B model, but it can run the streaming
kernel standalone at the real 7B shape (head_dim=128, 28 query heads, 4 KV
heads) against synthetic K/V -- enough to see whether the kernel's own
scaling behavior (not model-level plumbing) explains the regression.

Two things to separate:
1. Does per-call time scale with total_tokens the way a bandwidth/compute
   bound kernel should (roughly linear), or is there a large fixed cost per
   launch (which would mean fewer, larger groups should help -- already
   disproved on the real A40/7B run, but worth confirming the shape here)?
2. How does our kernel's absolute time compare to a dense reference
   (F.scaled_dot_product_attention) at the exact same shape? If even the
   best block_kv is far behind dense at every total_tokens, the gap is
   structural (grid size = batch*num_query_heads = 28 programs regardless
   of total_tokens -- far fewer than the GPU's SM count), not a tuning
   question.
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from pager_hf.streaming_attention import (
    streaming_attention_finalize,
    streaming_attention_state_init,
    streaming_attention_step,
)

# Real Qwen2.5-7B-Instruct shape.
NUM_QUERY_HEADS = 28
NUM_KV_HEADS = 4
HEAD_DIM = 128
N_REP = NUM_QUERY_HEADS // NUM_KV_HEADS


def time_cuda(fn, *, warmup=3, reps=20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps  # ms/call


def streaming_call(q, k, v, block_kv: int):
    import pager_hf.streaming_attention as sa

    old = sa._BLOCK_KV
    sa._BLOCK_KV = block_kv
    try:
        m, l, acc = streaming_attention_state_init(1, NUM_QUERY_HEADS, HEAD_DIM, q.device)
        streaming_attention_step(q, k, v, m, l, acc)
        streaming_attention_finalize(acc, l, q.dtype)
    finally:
        sa._BLOCK_KV = old


def dense_call_sdpa(q, k, v):
    # q: [1, num_query_heads, head_dim] -> [1, num_query_heads, 1, head_dim]
    qd = q.unsqueeze(2)
    kd = k.repeat_interleave(N_REP, dim=1)
    vd = v.repeat_interleave(N_REP, dim=1)
    F.scaled_dot_product_attention(qd, kd, vd)


def dense_call_eager(q, k, v):
    """Mirrors what transformers' attn_implementation="eager" actually runs: batched
    torch.matmul (real cuBLAS GEMM, tensor-core-eligible on Ampere+) instead of SDPA's
    math fallback -- the real competitor for the non-streaming reload-all path."""
    qd = q.unsqueeze(2)  # [1, num_query_heads, 1, head_dim]
    kd = k.repeat_interleave(N_REP, dim=1)  # [1, num_query_heads, total_tokens, head_dim]
    vd = v.repeat_interleave(N_REP, dim=1)
    scale = 1.0 / (HEAD_DIM**0.5)
    scores = torch.matmul(qd, kd.transpose(-2, -1)) * scale  # [1, num_query_heads, 1, total_tokens]
    probs = torch.softmax(scores, dim=-1)
    torch.matmul(probs, vd)  # [1, num_query_heads, 1, head_dim]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(device)}, dtype: {dtype}")
    print(f"shape: num_query_heads={NUM_QUERY_HEADS} num_kv_heads={NUM_KV_HEADS} head_dim={HEAD_DIM}")
    print()

    token_counts = [64, 256, 1024, 2048, 4096]
    block_kv_candidates = [32, 64, 128, 256]

    header = f"{'total_tokens':>12} | {'sdpa_ms':>9} | {'eager_ms':>9} |" + "".join(
        f" bkv={bkv:>4}_ms |" for bkv in block_kv_candidates
    )
    print(header)
    print("-" * len(header))

    for total_tokens in token_counts:
        q = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=dtype, device=device)
        k = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=dtype, device=device)
        v = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=dtype, device=device)

        sdpa_ms = time_cuda(lambda: dense_call_sdpa(q, k, v))
        eager_ms = time_cuda(lambda: dense_call_eager(q, k, v))
        row = f"{total_tokens:>12} | {sdpa_ms:>9.4f} | {eager_ms:>9.4f} |"
        for bkv in block_kv_candidates:
            if bkv > total_tokens:
                row += f" {'--':>10} |"
                continue
            try:
                ms = time_cuda(lambda bkv=bkv: streaming_call(q, k, v, bkv))
                row += f" {ms:>10.4f} |"
            except Exception as e:  # e.g. OutOfResources on GPUs with small shared memory (Pascal)
                row += f" {'ERR':>10} |"
        print(row + ("" if "ERR" not in row else "  (ERR = kernel launch failed, likely shared-mem limit)"))
        continue
        print(row)


if __name__ == "__main__":
    main()
