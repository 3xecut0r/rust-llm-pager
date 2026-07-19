from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import torch
from config import (
    MAX_LENGTH,
    MODEL_NAME,
    PROMOTE_MARGIN,
    RAM_BUDGET,
    RAM_PROMOTE_MARGIN,
    REBALANCE_INTERVAL,
    RECENT_WINDOW,
    TOKENS_PER_BLOCK,
    VRAM_BUDGET,
)
from real_kv_utils import (
    build_prompt,
    extract_last_query_block_attention,
    extract_single_block_from_past,
    format_block_list,
    get_legacy_past_key_values,
    real_past_to_blocks,
    reconstruct_past_from_store,
    split_full_blocks_and_tail,
)
from torch_kv_block_store import KVBlockStore
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

import pager

GENERATE_TOKENS = 64

POLICIES = ["recent_only", "sinks_recent", "heavy_hitter", "sinks_heavy_hitter"]


def parse_args() -> argparse.Namespace:
    """Parse --tokens and --quiet."""
    parser = argparse.ArgumentParser(description="Persistent CPU KV paging policy comparison")
    parser.add_argument(
        "--tokens",
        type=int,
        default=GENERATE_TOKENS,
        help="number of tokens to generate per policy (default: %(default)s)",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-policy debug output, only print the final table"
    )
    return parser.parse_args()


@dataclass
class PolicyRunResult:
    policy: str
    same_token_ids: bool
    total_new_blocks: int
    final_num_blocks: int
    total_gpu_to_cpu_mb: float
    total_cpu_to_gpu_mb: float
    total_gpu_to_cpu_copies: int
    total_cpu_to_gpu_copies: int
    mean_attention_in_gpu: float
    min_attention_in_gpu: float
    max_attention_in_gpu: float
    final_gpu_blocks: str
    final_cpu_blocks: str
    added_block_events: str


def greedy_baseline_generate(*, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int) -> list[int]:
    """Generate steps tokens the plain way, with the full KV cache always resident on GPU."""
    next_input_id = input_ids[:, -1:]
    current_attention_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True, output_attentions=False
        )

    current_cache = outputs.past_key_values
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_attention_mask,
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )

            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            generated.append(int(next_token_id.item()))

            current_cache = outputs.past_key_values
            next_input_id = next_token_id

            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )

    return generated


def build_initial_store(past_key_values) -> tuple[KVBlockStore, int, int, list[tuple[torch.Tensor, torch.Tensor]], int]:
    """Build the persistent KV store from the prefix's past_key_values."""
    full_past, tail_past, full_tokens = split_full_blocks_and_tail(past_key_values, tokens_per_block=TOKENS_PER_BLOCK)

    kv_blocks = real_past_to_blocks(full_past, tokens_per_block=TOKENS_PER_BLOCK, verbose=False)

    store = KVBlockStore(tokens_per_block=TOKENS_PER_BLOCK)

    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    num_layers = len(full_past)
    num_blocks = len(kv_blocks)

    return store, num_layers, num_blocks, tail_past, full_tokens


def append_new_full_blocks_if_needed(
    *, store: KVBlockStore, past_key_values, known_num_blocks: int
) -> tuple[int, list[tuple[torch.Tensor, torch.Tensor]], int, list[int]]:
    """Register any newly completed KV blocks in the store."""
    full_past, tail_past, full_tokens = split_full_blocks_and_tail(past_key_values, tokens_per_block=TOKENS_PER_BLOCK)
    new_num_blocks = full_tokens // TOKENS_PER_BLOCK

    added_block_ids = list(range(known_num_blocks, new_num_blocks))
    for block_id in added_block_ids:
        store.put_gpu(
            block_id, *extract_single_block_from_past(full_past, block_id=block_id, tokens_per_block=TOKENS_PER_BLOCK)
        )

    return new_num_blocks, tail_past, full_tokens, added_block_ids


def reload_all_blocks_for_forward(*, store: KVBlockStore, device: torch.device, num_blocks: int) -> tuple[int, int]:
    """Make sure every block is back on GPU before the next forward pass."""
    before = store.summary()
    for block_id in range(num_blocks):
        store.ensure_gpu(block_id, device)
    after = store.summary()

    return (
        after["cpu_to_gpu_bytes"] - before["cpu_to_gpu_bytes"],
        after["cpu_to_gpu_copies"] - before["cpu_to_gpu_copies"],
    )


def run_policy(
    *,
    policy: str,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    baseline_generated: list[int],
    generate_tokens: int,
) -> PolicyRunResult:
    """Run the persistent paging loop once for one policy and collect its stats."""
    device = input_ids.device

    next_input_id = input_ids[:, -1:]
    current_attention_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True, output_attentions=True
        )

    current_past = get_legacy_past_key_values(outputs)

    store, num_layers, num_blocks, tail_past, _ = build_initial_store(current_past)

    p = pager.PyPager(
        VRAM_BUDGET, RAM_BUDGET, RECENT_WINDOW, REBALANCE_INTERVAL, PROMOTE_MARGIN, RAM_PROMOTE_MARGIN, policy
    )

    generated: list[int] = []

    total_gpu_to_cpu_bytes = 0
    total_cpu_to_gpu_bytes = 0
    total_gpu_to_cpu_copies = 0
    total_cpu_to_gpu_copies = 0
    total_new_blocks = 0

    attention_in_gpu_values: list[float] = []
    added_block_events: list[str] = []

    for step in range(1, generate_tokens + 1):
        reload_bytes, reload_copies = reload_all_blocks_for_forward(store=store, device=device, num_blocks=num_blocks)

        current_past_for_forward = reconstruct_past_from_store(
            store=store, num_layers=num_layers, num_blocks=num_blocks, tail_past=tail_past
        )

        cache = DynamicCache.from_legacy_cache(tuple(current_past_for_forward))

        with torch.inference_mode():
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_attention_mask,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
            )

        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        generated.append(int(next_token_id.item()))

        current_past = get_legacy_past_key_values(outputs)

        old_num_blocks = num_blocks

        num_blocks, tail_past, _, added_block_ids = append_new_full_blocks_if_needed(
            store=store, past_key_values=current_past, known_num_blocks=old_num_blocks
        )

        if added_block_ids:
            added_block_events.append(f"step {step}: {added_block_ids}")

        total_new_blocks += len(added_block_ids)

        block_attention = extract_last_query_block_attention(
            outputs, tokens_per_block=TOKENS_PER_BLOCK, num_blocks=num_blocks
        )

        query_block = num_blocks - 1

        summary_before_placement = store.summary()

        p.on_step(query_block, 0, block_attention)

        if added_block_ids:
            p.force_rebalance(query_block)

        store.apply_tiers(p.tiers(), device)

        summary_after_placement = store.summary()

        total_gpu_to_cpu_bytes += (
            summary_after_placement["gpu_to_cpu_bytes"] - summary_before_placement["gpu_to_cpu_bytes"]
        )
        total_cpu_to_gpu_bytes += reload_bytes
        total_gpu_to_cpu_copies += (
            summary_after_placement["gpu_to_cpu_copies"] - summary_before_placement["gpu_to_cpu_copies"]
        )
        total_cpu_to_gpu_copies += reload_copies

        real_attention_in_gpu = sum(
            block_attention[block_id] for block_id in store.gpu_block_ids() if block_id < len(block_attention)
        )
        attention_in_gpu_values.append(real_attention_in_gpu)

        next_input_id = next_token_id

        current_attention_mask = torch.cat(
            [
                current_attention_mask,
                torch.ones(
                    (current_attention_mask.shape[0], 1),
                    dtype=current_attention_mask.dtype,
                    device=current_attention_mask.device,
                ),
            ],
            dim=1,
        )

    return PolicyRunResult(
        policy=policy,
        same_token_ids=baseline_generated == generated,
        total_new_blocks=total_new_blocks,
        final_num_blocks=num_blocks,
        total_gpu_to_cpu_mb=total_gpu_to_cpu_bytes / 1_000_000,
        total_cpu_to_gpu_mb=total_cpu_to_gpu_bytes / 1_000_000,
        total_gpu_to_cpu_copies=total_gpu_to_cpu_copies,
        total_cpu_to_gpu_copies=total_cpu_to_gpu_copies,
        mean_attention_in_gpu=sum(attention_in_gpu_values) / len(attention_in_gpu_values),
        min_attention_in_gpu=min(attention_in_gpu_values),
        max_attention_in_gpu=max(attention_in_gpu_values),
        final_gpu_blocks=format_block_list(store.gpu_block_ids()),
        final_cpu_blocks=format_block_list(store.cpu_block_ids()),
        added_block_events="; ".join(added_block_events),
    )


def write_results_csv(results: list[PolicyRunResult]) -> Path:
    """Write one CSV row per policy result and return the file path."""
    out_path = Path("bench/persistent_policy_compare_results.csv")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(PolicyRunResult.__dataclass_fields__.keys()))
        writer.writeheader()

        for result in results:
            writer.writerow(result.__dict__)

    return out_path


def write_results_markdown(results: list[PolicyRunResult], *, generate_tokens: int) -> Path:
    """Write a markdown table of policy results, ranked by mean attention kept on GPU."""
    out_path = Path("bench/persistent_policy_compare_results.md")

    ranked = sorted(results, key=lambda item: item.mean_attention_in_gpu, reverse=True)

    lines = [
        "# Persistent CPU KV paging: policy comparison",
        "",
        f"Model: `{MODEL_NAME}`  ",
        f"Generated tokens per policy: {generate_tokens}",
        "",
        "| Policy | Same as baseline | New blocks | Final blocks | Mean attn in VRAM | Min attn in VRAM | GPU->CPU MB | CPU->GPU MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for result in ranked:
        lines.append(
            f"| `{result.policy}` "
            f"| {result.same_token_ids} "
            f"| {result.total_new_blocks} "
            f"| {result.final_num_blocks} "
            f"| {result.mean_attention_in_gpu:.4f} "
            f"| {result.min_attention_in_gpu:.4f} "
            f"| {result.total_gpu_to_cpu_mb:.2f} "
            f"| {result.total_cpu_to_gpu_mb:.2f} |"
        )

    lines.append("")

    out_path.write_text("\n".join(lines))

    return out_path


def main() -> None:
    args = parse_args()
    generate_tokens = args.tokens
    quiet = args.quiet

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("tokens_per_block:", TOKENS_PER_BLOCK)
    print("generate_tokens:", generate_tokens)
    print("policies:", POLICIES)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager", torch_dtype=torch.float16).to(
        device
    )

    model.eval()

    prompt = build_prompt()

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("prompt_seq_len:", input_ids.shape[-1])

    baseline_generated = greedy_baseline_generate(
        model=model, input_ids=input_ids, attention_mask=attention_mask, steps=generate_tokens
    )

    results: list[PolicyRunResult] = []

    for policy in POLICIES:
        if not quiet:
            print(f"\nRunning policy: {policy}")

        result = run_policy(
            policy=policy,
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            attention_mask=attention_mask,
            baseline_generated=baseline_generated,
            generate_tokens=generate_tokens,
        )
        results.append(result)

        if not quiet:
            print("same_token_ids:", result.same_token_ids)
            print("total_new_blocks:", result.total_new_blocks)
            print("final_num_blocks:", result.final_num_blocks)
            print("mean_attention_in_gpu:", f"{result.mean_attention_in_gpu:.4f}")
            print("min_attention_in_gpu:", f"{result.min_attention_in_gpu:.4f}")
            print("gpu_to_cpu_mb:", f"{result.total_gpu_to_cpu_mb:.2f}")
            print("cpu_to_gpu_mb:", f"{result.total_cpu_to_gpu_mb:.2f}")
            print("final_gpu_blocks:", result.final_gpu_blocks)
            print("added_block_events:", result.added_block_events)

        assert result.same_token_ids
        assert result.total_new_blocks >= 3
        assert result.final_num_blocks >= 12

    csv_path = write_results_csv(results)
    markdown_path = write_results_markdown(results, generate_tokens=generate_tokens)

    print("\nPersistent policy comparison")
    print("----------------------------")

    for result in sorted(results, key=lambda item: item.mean_attention_in_gpu, reverse=True):
        print(
            f"{result.policy:20s} "
            f"mean_attn={result.mean_attention_in_gpu:.4f} "
            f"min_attn={result.min_attention_in_gpu:.4f} "
            f"gpu_to_cpu_mb={result.total_gpu_to_cpu_mb:.2f} "
            f"cpu_to_gpu_mb={result.total_cpu_to_gpu_mb:.2f}"
        )

    print("\nresults_csv:", csv_path)
    print("results_markdown:", markdown_path)
    print("\nOK: persistent CPU paging policy comparison completed.")


if __name__ == "__main__":
    main()
