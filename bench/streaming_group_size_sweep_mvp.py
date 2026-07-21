from __future__ import annotations

import argparse
import time

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

# Same constrained budget as streaming_paged_model_peak_memory_mvp.py, so
# blocks genuinely offload between steps and the sweep measures a real effect.
VRAM_BUDGET = 100_000_000
NEW_TOKENS = 6
GROUP_SIZES = [4, 8, 16, 32, 64, 128]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep streaming_group_size_blocks for peak memory and wall-clock time."
    )
    parser.add_argument("--context-tokens", type=int, default=5000)
    parser.add_argument("--policy", type=str, default="sinks_heavy_hitter")
    return parser.parse_args()


def run(*, model, policy: str, group_size: int, input_ids, attention_mask, device) -> tuple[list[int], float, float]:
    """Prime (max_new_tokens=0), reset peak stats, then time+measure the decode loop."""
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
        use_streaming_attention=True,
        streaming_group_size_blocks=group_size,
    )

    paged_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=0)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    started = time.perf_counter()
    generated = paged_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=NEW_TOKENS)
    torch.cuda.synchronize(device)
    elapsed_s = time.perf_counter() - started

    peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000
    return generated, peak_mb, elapsed_s


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
    print("group_sizes:", GROUP_SIZES)

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

    reference_ids = None
    rows = []

    for group_size in GROUP_SIZES:
        print(f"\nRunning group_size={group_size}...")
        generated, peak_mb, elapsed_s = run(
            model=model,
            policy=args.policy,
            group_size=group_size,
            input_ids=input_ids,
            attention_mask=attention_mask,
            device=device,
        )
        torch.cuda.empty_cache()

        if reference_ids is None:
            reference_ids = generated
        same_ids = generated == reference_ids

        rows.append((group_size, peak_mb, elapsed_s, same_ids))
        print(f"  peak_gpu_mb={peak_mb:.2f} decode_wall_s={elapsed_s:.3f} same_token_ids={same_ids}")

    print("\nSweep summary")
    print("-------------")
    print(f"{'group_size':>10s} {'peak_mb':>10s} {'decode_s':>10s} {'same_ids':>9s}")
    for group_size, peak_mb, elapsed_s, same_ids in rows:
        print(f"{group_size:>10d} {peak_mb:>10.2f} {elapsed_s:>10.3f} {str(same_ids):>9s}")

    if not all(same_ids for _, _, _, same_ids in rows):
        print("\nFAIL: generation diverged across group sizes -- group size should not affect exact-attention output.")
        raise SystemExit(1)

    best_by_memory = min(rows, key=lambda row: row[1])
    best_by_speed = min(rows, key=lambda row: row[2])
    print(f"\nbest_by_memory: group_size={best_by_memory[0]} ({best_by_memory[1]:.2f} MB)")
    print(f"best_by_speed:  group_size={best_by_speed[0]} ({best_by_speed[2]:.3f} s)")

    print("\nOK: same_token_ids held across every group size, as expected for exact (not approximate) attention.")


if __name__ == "__main__":
    main()
