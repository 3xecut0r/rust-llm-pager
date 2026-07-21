"""Where does decode-step time actually go in the real streaming path: the
attention kernel itself, or the per-block KV gather loop in
PagedModel._gather_layer_kv_group?

The tl.dot kernel rewrite made the raw kernel faster than dense attention in
isolation (see bench/streaming_attention_kernel_scaling_probe.py), but the
full-model scale test (bench/streaming_paged_model_scale_test_mvp.py) barely
moved. This instruments the real PagedModel/KVBlockStore path directly --
monkey-patching _gather_layer_kv_group and streaming_attention_step to
accumulate elapsed CUDA time and call counts -- to find out which one is
actually eating the time, on the real 7B/8000-token workload.
"""

from __future__ import annotations

import time

import torch
from real_kv_vram_savings_proof_mvp import build_long_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

import pager_hf.paged_model as paged_model_module
import pager_hf.streaming_attention as sa
from pager_hf import PagedModel

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONTEXT_TOKENS = 8000
TOKENS_PER_BLOCK = 16
NEW_TOKENS = 4
RAM_BUDGET = 8_000_000_000
VRAM_BUDGET = 2_000_000_000

_stats = {"gather_s": 0.0, "gather_calls": 0, "kernel_s": 0.0, "kernel_calls": 0}

_orig_gather = paged_model_module.PagedModel._gather_layer_kv_group
_orig_step = sa.streaming_attention_step


def _timed_gather(self, *args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = _orig_gather(self, *args, **kwargs)
    torch.cuda.synchronize()
    _stats["gather_s"] += time.perf_counter() - start
    _stats["gather_calls"] += 1
    return result


def _timed_step(*args, **kwargs):
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = _orig_step(*args, **kwargs)
    torch.cuda.synchronize()
    _stats["kernel_s"] += time.perf_counter() - start
    _stats["kernel_calls"] += 1
    return result


paged_model_module.PagedModel._gather_layer_kv_group = _timed_gather
sa.streaming_attention_step = _timed_step
paged_model_module.streaming_attention_step = _timed_step


def main() -> None:
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL, attn_implementation="eager", torch_dtype=torch.bfloat16).to(
        device
    )
    model.eval()

    prompt = build_long_prompt(tokenizer, CONTEXT_TOKENS)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=CONTEXT_TOKENS)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    print("actual_context_tokens:", input_ids.shape[-1])

    paged_model = PagedModel(
        model,
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        policy="recent_only",
        tokens_per_block=TOKENS_PER_BLOCK,
        use_streaming_attention=True,
        streaming_group_size_blocks=64,
    )

    print("priming...")
    paged_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=0)
    torch.cuda.synchronize(device)

    for key in _stats:
        _stats[key] = 0.0 if "s" in key else 0

    print(f"decoding {NEW_TOKENS} tokens, instrumented...")
    started = time.perf_counter()
    paged_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=NEW_TOKENS)
    torch.cuda.synchronize(device)
    total_s = time.perf_counter() - started

    print("\nProfile summary")
    print("---------------")
    print(f"total_decode_s ({NEW_TOKENS} tokens):", f"{total_s:.3f}")
    print(f"gather_s:  {_stats['gather_s']:.3f}  ({_stats['gather_calls']} calls, {_stats['gather_s']/total_s*100:.1f}% of total)")
    print(f"kernel_s:  {_stats['kernel_s']:.3f}  ({_stats['kernel_calls']} calls, {_stats['kernel_s']/total_s*100:.1f}% of total)")
    other_s = total_s - _stats["gather_s"] - _stats["kernel_s"]
    print(f"other_s:   {other_s:.3f}  ({other_s/total_s*100:.1f}% of total -- RoPE, o_proj, MLP, norm, tail cat, etc.)")


if __name__ == "__main__":
    main()
