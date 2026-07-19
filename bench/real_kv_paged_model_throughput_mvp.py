from __future__ import annotations

import argparse
import time

import torch
from config import MODEL_NAME, RAM_BUDGET, RECENT_WINDOW, TOKENS_PER_BLOCK, VRAM_BUDGET
from real_kv_vram_savings_proof_mvp import build_long_prompt, chunked_prefill
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel

POLICY = "recent_only"
CONTEXT_TOKENS = 2000
NEW_TOKENS = 32
PREFILL_CHUNK = 512


def parse_args() -> argparse.Namespace:
    """Parse --context-tokens, --new-tokens, and --policy."""
    parser = argparse.ArgumentParser(
        description="Measure steady-state decode throughput: unpaged baseline vs pager_hf.PagedModel."
    )
    parser.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument(
        "--policy", default=POLICY, choices=["recent_only", "sinks_recent", "heavy_hitter", "sinks_heavy_hitter"]
    )
    return parser.parse_args()


def timed_baseline_decode(*, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int):
    """Decode steps tokens one at a time, timing only the per-step decode loop (prefill excluded)."""
    next_input_id = input_ids[:, -1:]
    current_mask = attention_mask
    step_seconds: list[float] = []

    outputs = chunked_prefill(model, input_ids[:, :-1], attention_mask[:, :-1], PREFILL_CHUNK)
    current_cache = outputs.past_key_values

    with torch.inference_mode():
        for _ in range(steps):
            torch.cuda.synchronize()
            started = time.perf_counter()

            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_mask,
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )

            torch.cuda.synchronize()
            step_seconds.append(time.perf_counter() - started)

            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            current_cache = outputs.past_key_values
            next_input_id = next_token_id
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=current_mask.device),
                ],
                dim=1,
            )

    return step_seconds


def timed_paged_decode(*, paged_model: PagedModel, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int):
    """Prime + one warmup token (untimed), then decode `steps` more tokens with zero extra priming, timed as a whole."""
    warmup_token = paged_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1)

    extended_ids = torch.cat([input_ids, torch.tensor([warmup_token], device=input_ids.device)], dim=1)
    extended_mask = torch.ones_like(extended_ids)

    torch.cuda.synchronize()
    started = time.perf_counter()

    paged_model.generate(input_ids=extended_ids, attention_mask=extended_mask, max_new_tokens=steps)

    torch.cuda.synchronize()
    return time.perf_counter() - started


def main() -> None:
    args = parse_args()
    context_tokens = args.context_tokens
    new_tokens = args.new_tokens
    policy = args.policy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", policy)
    print("context_tokens (target):", context_tokens)
    print("new_tokens (timed, after 1 untimed warmup token):", new_tokens)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()

    prompt = build_long_prompt(tokenizer, context_tokens)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("actual_context_tokens:", input_ids.shape[-1])

    print("\nTiming unpaged baseline decode...")
    baseline_step_seconds = timed_baseline_decode(
        model=model, input_ids=input_ids, attention_mask=attention_mask, steps=new_tokens
    )
    baseline_total_seconds = sum(baseline_step_seconds)

    print("\nTiming pager_hf.PagedModel decode...")
    paged_model = PagedModel(
        model,
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        recent_window=RECENT_WINDOW,
        policy=policy,
        tokens_per_block=TOKENS_PER_BLOCK,
        prefill_chunk_tokens=PREFILL_CHUNK,
    )
    paged_total_seconds = timed_paged_decode(
        paged_model=paged_model, input_ids=input_ids, attention_mask=attention_mask, steps=new_tokens
    )

    print("\nThroughput summary")
    print("-------------------")
    print(f"baseline_total_sec: {baseline_total_seconds:.4f}")
    print(f"baseline_mean_step_ms: {1000 * baseline_total_seconds / new_tokens:.2f}")
    print(f"baseline_tokens_per_sec: {new_tokens / baseline_total_seconds:.2f}")
    print(f"paged_total_sec: {paged_total_seconds:.4f}")
    print(f"paged_mean_step_ms: {1000 * paged_total_seconds / new_tokens:.2f}")
    print(f"paged_tokens_per_sec: {new_tokens / paged_total_seconds:.2f}")
    print(f"paged_slowdown_factor: {paged_total_seconds / baseline_total_seconds:.2f}x")

    print("\nOK: throughput measured for both baseline and paged decode.")


if __name__ == "__main__":
    main()
