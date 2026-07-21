from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

HEAD_DIM = 64
NUM_HEADS = 8


def streaming_attention(q: torch.Tensor, kv_chunks: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """
    Online-softmax attention for a single query position, fed one KV chunk at a
    time instead of one big concatenated tensor.

    q: [heads, head_dim]
    kv_chunks: list of (key, value), each [heads, chunk_len, head_dim]
    Returns: [heads, head_dim], same dtype as q.

    Mathematically exact (not an approximation) reorganization of softmax
    attention -- accumulates in fp32 regardless of input dtype, matching the
    FlashAttention convention, so results are close-but-not-bit-identical to a
    dense softmax computed some other way, not because the math differs.
    """
    heads, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    m = torch.full((heads, 1), float("-inf"), dtype=torch.float32, device=q.device)
    l = torch.zeros((heads, 1), dtype=torch.float32, device=q.device)
    acc = torch.zeros((heads, head_dim), dtype=torch.float32, device=q.device)

    q32 = q.to(torch.float32)

    for key_chunk, value_chunk in kv_chunks:
        scores = torch.einsum("hd,hcd->hc", q32, key_chunk.to(torch.float32)) * scale

        chunk_max = scores.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m, chunk_max)
        rescale = torch.exp(m - m_new)

        acc = acc * rescale
        l = l * rescale

        p = torch.exp(scores - m_new)
        acc = acc + torch.einsum("hc,hcd->hd", p, value_chunk.to(torch.float32))
        l = l + p.sum(dim=-1, keepdim=True)

        m = m_new

    return (acc / l).to(q.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand grouped-query KV heads to match the query head count.

    x: [kv_heads, seq_len, head_dim] -> [kv_heads * n_rep, seq_len, head_dim].
    Purely per-position (no mixing across seq_len), so applying it to a small
    chunk gives the same result as applying it to the full concatenated
    tensor and then slicing out that chunk.
    """
    if n_rep == 1:
        return x

    kv_heads, seq_len, head_dim = x.shape
    return x[:, None, :, :].expand(kv_heads, n_rep, seq_len, head_dim).reshape(kv_heads * n_rep, seq_len, head_dim)


def streaming_attention_gqa(
    q: torch.Tensor, kv_chunks: list[tuple[torch.Tensor, torch.Tensor]], *, n_rep: int
) -> torch.Tensor:
    """streaming_attention, but each chunk's KV heads are expanded to the query head count first."""
    expanded_chunks = [(repeat_kv(key, n_rep), repeat_kv(value, n_rep)) for key, value in kv_chunks]
    return streaming_attention(q, expanded_chunks)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Split the last dim in half and swap-negate, the standard RoPE building block."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, position_ids: torch.Tensor, *, theta: float = 10000.0) -> torch.Tensor:
    """
    Apply rotary position embedding at the given absolute positions.

    x: [heads, seq_len, head_dim], position_ids: [seq_len] -> rotated x, same shape/dtype.
    """
    head_dim = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device) / head_dim))
    freqs = torch.einsum("s,d->sd", position_ids.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos()[None, :, :], emb.sin()[None, :, :]

    return (x.to(torch.float32) * cos + rotate_half(x.to(torch.float32)) * sin).to(x.dtype)


def dense_attention_reference(q: torch.Tensor, kv_chunks: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Standard scaled_dot_product_attention over the full concatenated context, for comparison."""
    key_full = torch.cat([key for key, _ in kv_chunks], dim=1)
    value_full = torch.cat([value for _, value in kv_chunks], dim=1)

    query = q.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, head_dim]
    key = key_full.unsqueeze(0)  # [1, heads, total_len, head_dim]
    value = value_full.unsqueeze(0)

    out = F.scaled_dot_product_attention(query, key, value, is_causal=False)
    return out.squeeze(0).squeeze(1)  # [heads, head_dim]


def make_random_chunks(
    *, total_tokens: int, chunk_size: int, heads: int, head_dim: int, dtype: torch.dtype, device: torch.device
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split total_tokens worth of random K/V into chunks of chunk_size (last chunk may be shorter)."""
    chunks = []
    for start in range(0, total_tokens, chunk_size):
        chunk_len = min(chunk_size, total_tokens - start)
        key_chunk = torch.randn(heads, chunk_len, head_dim, dtype=dtype, device=device)
        value_chunk = torch.randn(heads, chunk_len, head_dim, dtype=dtype, device=device)
        chunks.append((key_chunk, value_chunk))
    return chunks


def run_case(
    *, name: str, total_tokens: int, chunk_size: int, dtype: torch.dtype, device: torch.device, rtol: float, atol: float
) -> bool:
    """Compare streaming_attention against the dense reference for one configuration."""
    torch.manual_seed(0)

    q = torch.randn(NUM_HEADS, HEAD_DIM, dtype=dtype, device=device)
    kv_chunks = make_random_chunks(
        total_tokens=total_tokens, chunk_size=chunk_size, heads=NUM_HEADS, head_dim=HEAD_DIM, dtype=dtype, device=device
    )

    streaming_out = streaming_attention(q, kv_chunks)
    dense_out = dense_attention_reference(q, kv_chunks).to(dtype)

    max_diff = (streaming_out.to(torch.float32) - dense_out.to(torch.float32)).abs().max().item()
    close = torch.allclose(streaming_out, dense_out, rtol=rtol, atol=atol)

    print(f"{name:35s} chunks={len(kv_chunks):4d} dtype={str(dtype):15s} max_diff={max_diff:.6f} match={close}")
    return close


def run_order_independence_case(
    *, total_tokens: int, chunk_size: int, dtype: torch.dtype, device: torch.device
) -> bool:
    """Online softmax must give the same result regardless of chunk feed order."""
    torch.manual_seed(1)

    q = torch.randn(NUM_HEADS, HEAD_DIM, dtype=dtype, device=device)
    kv_chunks = make_random_chunks(
        total_tokens=total_tokens, chunk_size=chunk_size, heads=NUM_HEADS, head_dim=HEAD_DIM, dtype=dtype, device=device
    )

    forward_out = streaming_attention(q, kv_chunks)

    shuffled = kv_chunks.copy()
    random.Random(2).shuffle(shuffled)
    shuffled_out = streaming_attention(q, shuffled)

    close = torch.allclose(forward_out, shuffled_out, rtol=1e-4, atol=1e-4)
    max_diff = (forward_out.to(torch.float32) - shuffled_out.to(torch.float32)).abs().max().item()
    print(
        f"{'order_independence':35s} chunks={len(kv_chunks):4d} dtype={str(dtype):15s} max_diff={max_diff:.6f} match={close}"
    )
    return close


def run_gqa_rope_case(
    *,
    total_tokens: int,
    chunk_size: int,
    num_query_heads: int,
    num_kv_heads: int,
    dtype: torch.dtype,
    device: torch.device,
) -> bool:
    """
    Combine GQA head expansion with RoPE already baked into historical chunks,
    matching Qwen2/Llama's actual conventions: K chunks carry RoPE from their
    original absolute positions (never reapplied), only the live query token's
    RoPE uses its current position.
    """
    torch.manual_seed(3)
    n_rep = num_query_heads // num_kv_heads

    key_full = torch.randn(num_kv_heads, total_tokens, HEAD_DIM, dtype=dtype, device=device)
    value_full = torch.randn(num_kv_heads, total_tokens, HEAD_DIM, dtype=dtype, device=device)
    key_full = apply_rope(key_full, torch.arange(total_tokens, device=device))

    kv_chunks = [
        (key_full[:, start : start + chunk_size, :], value_full[:, start : start + chunk_size, :])
        for start in range(0, total_tokens, chunk_size)
    ]

    query = torch.randn(num_query_heads, 1, HEAD_DIM, dtype=dtype, device=device)
    query = apply_rope(query, torch.tensor([total_tokens], device=device)).squeeze(1)  # [heads, head_dim]

    streaming_out = streaming_attention_gqa(query, kv_chunks, n_rep=n_rep)

    dense_out = dense_attention_reference(query, [(repeat_kv(key_full, n_rep), repeat_kv(value_full, n_rep))]).to(dtype)

    max_diff = (streaming_out.to(torch.float32) - dense_out.to(torch.float32)).abs().max().item()
    close = torch.allclose(streaming_out, dense_out, rtol=2e-2, atol=2e-2)
    print(f"{'gqa_rope':35s} chunks={len(kv_chunks):4d} dtype={str(dtype):15s} max_diff={max_diff:.6f} match={close}")
    return close


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print()

    results = []

    results.append(
        run_case(
            name="single_chunk_fp32",
            total_tokens=128,
            chunk_size=128,
            dtype=torch.float32,
            device=device,
            rtol=1e-5,
            atol=1e-5,
        )
    )
    results.append(
        run_case(
            name="few_chunks_fp32",
            total_tokens=200,
            chunk_size=16,
            dtype=torch.float32,
            device=device,
            rtol=1e-4,
            atol=1e-4,
        )
    )
    results.append(
        run_case(
            name="many_chunks_fp32",
            total_tokens=2000,
            chunk_size=16,
            dtype=torch.float32,
            device=device,
            rtol=1e-3,
            atol=1e-3,
        )
    )
    results.append(
        run_case(
            name="few_chunks_fp16",
            total_tokens=200,
            chunk_size=16,
            dtype=torch.float16,
            device=device,
            rtol=2e-2,
            atol=2e-2,
        )
    )
    results.append(
        # Adversarial: many one-token chunks in fp16, the case most likely to expose numerical instability.
        run_case(
            name="adversarial_one_token_chunks_fp16",
            total_tokens=64,
            chunk_size=1,
            dtype=torch.float16,
            device=device,
            rtol=2e-2,
            atol=2e-2,
        )
    )
    results.append(run_order_independence_case(total_tokens=300, chunk_size=13, dtype=torch.float32, device=device))
    results.append(
        # 14 query heads / 2 KV heads, head_dim 64: matches Qwen2.5-0.5B-Instruct's real GQA shape.
        run_gqa_rope_case(
            total_tokens=200, chunk_size=16, num_query_heads=14, num_kv_heads=2, dtype=torch.float16, device=device
        )
    )
    results.append(
        run_gqa_rope_case(
            total_tokens=2000, chunk_size=16, num_query_heads=14, num_kv_heads=2, dtype=torch.float16, device=device
        )
    )

    print()
    if all(results):
        print(f"OK: streaming online-softmax attention matches dense attention in all {len(results)} cases.")
    else:
        print(f"FAIL: {results.count(False)}/{len(results)} cases did not match.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
