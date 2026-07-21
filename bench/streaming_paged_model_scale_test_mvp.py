from __future__ import annotations

import argparse
import time

import torch
from real_kv_vram_savings_proof_mvp import build_long_prompt, chunked_prefill
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel

# One-shot scale test for real hardware beyond the dev GTX 1050 Ti (4GB, Pascal):
# a real 7B+ model, real long context, real GPU with real VRAM headroom. Reuses
# the same correctness bar as every other bench script in this project
# (same_token_ids against an unpaged baseline), plus the same peak-memory and
# throughput measurements already proven on the small dev model, just at scale.

NEW_TOKENS = 8
TOKENS_PER_BLOCK = 16
RAM_BUDGET = 8_000_000_000


def greedy_baseline_generate(*, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int) -> list[int]:
    next_input_id = input_ids[:, -1:]
    current_mask = attention_mask

    outputs = chunked_prefill(model, input_ids[:, :-1], attention_mask[:, :-1], 512)
    current_cache = outputs.past_key_values
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_mask,
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(next_token_id.item()))
            current_cache = outputs.past_key_values
            next_input_id = next_token_id
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=current_mask.device),
                ],
                dim=1,
            )

    return generated


def run_paged(
    *,
    model,
    policy: str,
    use_streaming: bool,
    vram_budget: int,
    input_ids,
    attention_mask,
    device,
    group_size_blocks: int,
) -> tuple[list[int], float, float]:
    """Prime (max_new_tokens=0), reset peak stats + start a timer, then decode. Returns (ids, peak_mb, elapsed_s)."""
    paged_model = PagedModel(
        model,
        vram_budget=vram_budget,
        ram_budget=RAM_BUDGET,
        policy=policy,
        tokens_per_block=TOKENS_PER_BLOCK,
        use_streaming_attention=use_streaming,
        streaming_group_size_blocks=group_size_blocks,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming-attention scale test: real model, real long context.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--context-tokens", type=int, default=8000)
    parser.add_argument("--policy", type=str, default="sinks_heavy_hitter")
    parser.add_argument("--vram-budget-mb", type=int, default=2000, help="VRAM budget for the pager, in MB")
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16"],
        help="float16 can overflow to NaN on larger models at long context -- bfloat16 is the safe default here.",
    )
    parser.add_argument("--group-size-blocks", type=int, default=64, help="streaming_group_size_blocks for PagedModel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this test.")

    device = torch.device("cuda")
    total_vram_mb = torch.cuda.get_device_properties(device).total_memory / 1_000_000
    print("device:", torch.cuda.get_device_name(device))
    print("total_vram_mb:", f"{total_vram_mb:.0f}")
    print("model:", args.model)
    print("policy:", args.policy)
    print("context_tokens (target):", args.context_tokens)
    print("new_tokens:", NEW_TOKENS)
    print("vram_budget_mb (pager):", args.vram_budget_mb)
    print("dtype:", args.dtype)
    print("group_size_blocks:", args.group_size_blocks)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager", torch_dtype=dtype).to(device)
    model.eval()
    print("model_type:", model.config.model_type)
    print(
        "num_hidden_layers:",
        model.config.num_hidden_layers,
        "num_attention_heads:",
        model.config.num_attention_heads,
        "num_key_value_heads:",
        model.config.num_key_value_heads,
    )

    weights_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1_000_000
    print("model_weights_mb:", f"{weights_mb:.0f}")

    prompt = build_long_prompt(tokenizer, args.context_tokens)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.context_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    print("actual_context_tokens:", input_ids.shape[-1])

    print("\nRunning unpaged baseline (chunked prefill, full KV always resident)...")
    torch.cuda.reset_peak_memory_stats(device)
    baseline_ids = greedy_baseline_generate(
        model=model, input_ids=input_ids, attention_mask=attention_mask, steps=NEW_TOKENS
    )
    baseline_peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000
    torch.cuda.empty_cache()

    vram_budget = args.vram_budget_mb * 1_000_000

    print("Running non-streaming PagedModel (reload-all-then-attend)...")
    non_streaming_ids, non_streaming_peak_mb, non_streaming_s = run_paged(
        model=model,
        policy=args.policy,
        use_streaming=False,
        vram_budget=vram_budget,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
        group_size_blocks=args.group_size_blocks,
    )
    torch.cuda.empty_cache()

    print("Running streaming PagedModel (Triton, on by default)...")
    streaming_ids, streaming_peak_mb, streaming_s = run_paged(
        model=model,
        policy=args.policy,
        use_streaming=True,
        vram_budget=vram_budget,
        input_ids=input_ids,
        attention_mask=attention_mask,
        device=device,
        group_size_blocks=args.group_size_blocks,
    )

    print("\nScale test summary")
    print("-------------------")
    print("baseline_ids:        ", baseline_ids)
    print("non_streaming_ids:   ", non_streaming_ids)
    print("streaming_ids:       ", streaming_ids)
    print("streaming_matches_baseline:", streaming_ids == baseline_ids)
    print("streaming_matches_non_streaming:", streaming_ids == non_streaming_ids)
    print()
    print("baseline_peak_gpu_mb:      ", f"{baseline_peak_mb:.2f}")
    print("non_streaming_decode_peak_gpu_mb:", f"{non_streaming_peak_mb:.2f}")
    print("streaming_decode_peak_gpu_mb:    ", f"{streaming_peak_mb:.2f}")
    print(f"streaming_vs_baseline_peak_reduction: {100.0 * (1.0 - streaming_peak_mb / baseline_peak_mb):.1f}%")
    print()
    print(f"non_streaming_decode_s ({NEW_TOKENS} tokens):", f"{non_streaming_s:.3f}")
    print(f"streaming_decode_s ({NEW_TOKENS} tokens):    ", f"{streaming_s:.3f}")
    print(f"streaming_speedup_vs_non_streaming: {non_streaming_s / streaming_s:.2f}x")

    assert streaming_ids == baseline_ids, "streaming PagedModel diverged from the unpaged baseline"
    assert streaming_ids == non_streaming_ids, "streaming PagedModel diverged from the non-streaming PagedModel"

    print("\nOK: streaming PagedModel matches baseline exactly at real scale.")


if __name__ == "__main__":
    main()
