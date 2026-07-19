from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from config import (
    MAX_LENGTH,
    MODEL_NAME,
    POLICY,
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


def parse_args() -> argparse.Namespace:
    """Parse --tokens and --quiet."""
    parser = argparse.ArgumentParser(description="Persistent CPU KV paging MVP stress test")
    parser.add_argument(
        "--tokens", type=int, default=GENERATE_TOKENS, help="number of tokens to generate (default: %(default)s)"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-step debug output, only print the final summary"
    )
    return parser.parse_args()


def greedy_baseline_generate(*, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int) -> list[int]:
    """Generate steps tokens the plain way, with the full KV cache always resident on GPU."""
    prefix_input_ids = input_ids[:, :-1]
    prefix_attention_mask = attention_mask[:, :-1]

    next_input_id = input_ids[:, -1:]
    current_attention_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=prefix_input_ids, attention_mask=prefix_attention_mask, use_cache=True, output_attentions=False
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


def build_initial_store(
    past_key_values, *, verbose: bool = True
) -> tuple[KVBlockStore, int, int, list[tuple[torch.Tensor, torch.Tensor]], int]:
    """Build the persistent KV store from the prefix's past_key_values."""
    full_past, tail_past, full_tokens = split_full_blocks_and_tail(past_key_values, tokens_per_block=TOKENS_PER_BLOCK)

    kv_blocks = real_past_to_blocks(full_past, tokens_per_block=TOKENS_PER_BLOCK, verbose=verbose)

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


def write_results_json(results: dict) -> Path:
    """Write the run's summary dict to a JSON file and return its path."""
    out_path = Path("bench/persistent_cpu_paging_stress_results.json")
    out_path.write_text(json.dumps(results, indent=2))
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
    print("policy:", POLICY)
    print("tokens_per_block:", TOKENS_PER_BLOCK)
    print("generate_tokens:", generate_tokens)

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

    next_input_id = input_ids[:, -1:]
    current_attention_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True, output_attentions=True
        )

    current_past = get_legacy_past_key_values(outputs)

    store, num_layers, num_blocks, tail_past, full_tokens = build_initial_store(current_past, verbose=not quiet)

    if not quiet:
        print("\nInitial persistent KV store")
        print("---------------------------")
        print("num_layers:", num_layers)
        print("num_blocks:", num_blocks)
        print("full_tokens:", full_tokens)
        print("tail_tokens:", tail_past[0][0].shape[2])
        print("store_gpu_blocks:", format_block_list(store.gpu_block_ids()))
        print("store_cpu_blocks:", format_block_list(store.cpu_block_ids()))

    p = pager.PyPager(
        VRAM_BUDGET, RAM_BUDGET, RECENT_WINDOW, REBALANCE_INTERVAL, PROMOTE_MARGIN, RAM_PROMOTE_MARGIN, POLICY
    )

    generated: list[int] = []

    total_gpu_to_cpu_bytes = 0
    total_cpu_to_gpu_bytes = 0
    total_gpu_to_cpu_copies = 0
    total_cpu_to_gpu_copies = 0
    total_new_blocks = 0

    attention_in_gpu_values: list[float] = []

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

        token_id = int(next_token_id.item())
        generated.append(token_id)

        current_past = get_legacy_past_key_values(outputs)

        old_num_blocks = num_blocks

        num_blocks, tail_past, full_tokens, added_block_ids = append_new_full_blocks_if_needed(
            store=store, past_key_values=current_past, known_num_blocks=old_num_blocks
        )

        total_new_blocks += len(added_block_ids)

        block_attention = extract_last_query_block_attention(
            outputs, tokens_per_block=TOKENS_PER_BLOCK, num_blocks=num_blocks
        )

        query_block = num_blocks - 1

        summary_before_placement = store.summary()

        p.on_step(query_block, 0, block_attention)

        if added_block_ids:
            p.force_rebalance(query_block)

        movement = store.apply_tiers(p.tiers(), device)

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

        if not quiet:
            print(f"\nPersistent paging step {step}")
            print("-----------------------------")
            print("generated_token_id:", token_id)
            print("generated_token:", repr(tokenizer.decode([token_id])))
            print("full_tokens:", full_tokens)
            print("tail_tokens:", tail_past[0][0].shape[2])
            print("num_blocks:", num_blocks)
            print("added_blocks:", added_block_ids)
            print("moved_to_cpu:", format_block_list(movement["to_cpu"]))
            print("moved_to_gpu:", format_block_list(movement["to_gpu"]))
            print("reload_cpu_to_gpu_mb:", f"{reload_bytes / 1_000_000:.2f}")
            print("reload_cpu_to_gpu_copies:", reload_copies)
            print("pager_vram_blocks:", format_block_list(p.vram_block_ids()))
            print("store_gpu_blocks:", format_block_list(store.gpu_block_ids()))
            print("store_cpu_blocks:", format_block_list(store.cpu_block_ids()))
            print("real_attention_in_gpu:", f"{real_attention_in_gpu:.4f}")
            print("resident_gpu_mb:", f"{summary_after_placement['resident_gpu_bytes'] / 1_000_000:.2f}")
            print("resident_cpu_mb:", f"{summary_after_placement['resident_cpu_bytes'] / 1_000_000:.2f}")

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

    baseline_text = tokenizer.decode(baseline_generated)
    generated_text = tokenizer.decode(generated)

    print("\nPersistent CPU paging summary")
    print("-----------------------------")
    print("baseline_ids:", baseline_generated)
    print("paged_ids:", generated)
    print("same_token_ids:", baseline_generated == generated)
    print("baseline_text:", repr(baseline_text))
    print("paged_text:", repr(generated_text))
    print("total_new_blocks:", total_new_blocks)
    print("final_num_blocks:", num_blocks)
    print("total_gpu_to_cpu_mb:", f"{total_gpu_to_cpu_bytes / 1_000_000:.2f}")
    print("total_cpu_to_gpu_mb:", f"{total_cpu_to_gpu_bytes / 1_000_000:.2f}")
    print("total_gpu_to_cpu_copies:", total_gpu_to_cpu_copies)
    print("total_cpu_to_gpu_copies:", total_cpu_to_gpu_copies)
    print("mean_attention_in_gpu:", f"{sum(attention_in_gpu_values) / len(attention_in_gpu_values):.4f}")
    print("min_attention_in_gpu:", f"{min(attention_in_gpu_values):.4f}")
    print("max_attention_in_gpu:", f"{max(attention_in_gpu_values):.4f}")

    results = {
        "policy": POLICY,
        "generate_tokens": generate_tokens,
        "same_token_ids": baseline_generated == generated,
        "total_new_blocks": total_new_blocks,
        "final_num_blocks": num_blocks,
        "total_gpu_to_cpu_mb": total_gpu_to_cpu_bytes / 1_000_000,
        "total_cpu_to_gpu_mb": total_cpu_to_gpu_bytes / 1_000_000,
        "total_gpu_to_cpu_copies": total_gpu_to_cpu_copies,
        "total_cpu_to_gpu_copies": total_cpu_to_gpu_copies,
        "mean_attention_in_gpu": sum(attention_in_gpu_values) / len(attention_in_gpu_values),
        "min_attention_in_gpu": min(attention_in_gpu_values),
        "max_attention_in_gpu": max(attention_in_gpu_values),
        "final_gpu_blocks": store.gpu_block_ids(),
        "final_cpu_blocks": store.cpu_block_ids(),
        "baseline_ids": baseline_generated,
        "paged_ids": generated,
    }

    json_path = write_results_json(results)
    print("\nresults_json:", json_path)

    assert baseline_generated == generated

    assert total_new_blocks >= 3
    assert num_blocks >= 12
    assert baseline_generated == generated

    print("\nOK: persistent CPU KV paging loop matches baseline greedy generation.")


if __name__ == "__main__":
    main()
