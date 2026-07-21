from __future__ import annotations

import argparse

import torch
from config import (
    MODEL_NAME,
    PROMOTE_MARGIN,
    RAM_BUDGET,
    RAM_PROMOTE_MARGIN,
    REBALANCE_INTERVAL,
    RECENT_WINDOW,
    TOKENS_PER_BLOCK,
)
from real_kv_vram_savings_proof_mvp import build_long_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel

# Small VRAM budget so the pager actually offloads most blocks to CPU between
# steps -- the peak-memory win only shows up once residency is genuinely
# split, not when everything fits on GPU anyway.
VRAM_BUDGET = 100_000_000
NEW_TOKENS = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Peak GPU memory: streaming vs non-streaming PagedModel, real product."
    )
    parser.add_argument("--context-tokens", type=int, default=3000)
    parser.add_argument("--policy", type=str, default="sinks_heavy_hitter")
    return parser.parse_args()


def run(*, model, policy: str, use_streaming: bool, input_ids, attention_mask, device) -> tuple[list[int], float]:
    """Prime the session (max_new_tokens=0), reset peak stats, then decode -- isolating decode-loop peak from priming."""
    paged_model = PagedModel(
        model,
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        recent_window=RECENT_WINDOW,
        rebalance_interval=REBALANCE_INTERVAL,
        promote_margin=PROMOTE_MARGIN,
        ram_promote_margin=RAM_PROMOTE_MARGIN,
        policy=policy,
        tokens_per_block=TOKENS_PER_BLOCK,
        use_streaming_attention=use_streaming,
        streaming_group_size_blocks=16,
    )

    paged_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    generated = paged_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=NEW_TOKENS)
    peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000

    return generated, peak_mb


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")
    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", args.policy)
    print("context_tokens (target):", args.context_tokens)
    print("new_tokens:", NEW_TOKENS)
    print("vram_budget_mb:", VRAM_BUDGET / 1_000_000)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager", torch_dtype=torch.float16).to(
        device
    )
    model.eval()

    prompt = build_long_prompt(tokenizer, args.context_tokens)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.context_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    print("actual_context_tokens:", input_ids.shape[-1])

    print("\nRunning non-streaming PagedModel (reload-all-then-attend)...")
    non_streaming_ids, non_streaming_peak_mb = run(
        model=model,
        policy=args.policy,
        use_streaming=False,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
    )
    torch.cuda.empty_cache()

    print("Running streaming PagedModel (Triton, group-wise)...")
    streaming_ids, streaming_peak_mb = run(
        model=model,
        policy=args.policy,
        use_streaming=True,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
    )

    print("\nPeak memory summary")
    print("-------------------")
    print("non_streaming_ids:", non_streaming_ids)
    print("streaming_ids:    ", streaming_ids)
    print("same_token_ids:", non_streaming_ids == streaming_ids)
    print("non_streaming_decode_peak_gpu_mb:", f"{non_streaming_peak_mb:.2f}")
    print("streaming_decode_peak_gpu_mb:", f"{streaming_peak_mb:.2f}")
    print(f"decode_peak_reduction: {100.0 * (1.0 - streaming_peak_mb / non_streaming_peak_mb):.1f}%")

    assert non_streaming_ids == streaming_ids, "streaming PagedModel diverged from non-streaming PagedModel"

    print(
        "\nOK: streaming PagedModel matches non-streaming PagedModel exactly, with lower decode-loop peak GPU memory."
    )


if __name__ == "__main__":
    main()
