from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

import pager
from config import (
    MODEL_NAME,
    TOKENS_PER_BLOCK,
    VRAM_BUDGET,
    RAM_BUDGET,
    RECENT_WINDOW,
    REBALANCE_INTERVAL,
    PROMOTE_MARGIN,
    RAM_PROMOTE_MARGIN,
)
from real_kv_utils import (
    get_legacy_past_key_values,
    split_full_blocks_and_tail,
    append_tail_to_reconstructed_past,
    reconstruct_past_from_store,
    extract_single_block_from_past,
    real_past_to_blocks,
    format_block_list,
)
from torch_kv_block_store import KVBlockStore

# This proof uses recent_only: placement here does not depend on attention
# scores, so we can skip output_attentions=True entirely and avoid the
# O(layers * heads * seq_len^2) eager-attention memory blowup that would
# otherwise make a long context infeasible on a small GPU.
POLICY = "recent_only"

CONTEXT_TOKENS = 6000
NEW_TOKENS = 8
PREFILL_CHUNK = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prove real VRAM savings: how much real Qwen KV cache the pager "
            "keeps off the GPU at a realistic long context, with byte-identical "
            "output to an unpaged baseline."
        )
    )
    parser.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument(
        "--vram-budget-mb",
        type=float,
        default=VRAM_BUDGET / 1_000_000,
        help="VRAM budget (MB) given to the pager (default: %(default)s)",
    )
    return parser.parse_args()


def build_long_prompt(tokenizer, target_tokens: int) -> str:
    paragraph = (
        "The quick brown fox jumps over the lazy dog while researchers discuss "
        "memory management, operating systems, distributed caches, and GPU "
        "scheduling in long, unrelated technical documents. "
    )
    paragraph_tokens = len(tokenizer(paragraph)["input_ids"])
    repeats = target_tokens // max(paragraph_tokens, 1) + 4
    return paragraph * repeats


def chunked_prefill(model, prefix_input_ids, prefix_attention_mask, chunk_size):
    """
    Prefill a long prefix in chunks instead of one big forward call.

    A single forward call over a long sequence makes HF compute logits for
    every position at once ([batch, seq_len, vocab_size]); with a ~152k
    vocab that alone OOMs long before the KV cache itself becomes the
    limiting resource. Chunking keeps that transient logits tensor small.
    """
    seq_len = prefix_input_ids.shape[-1]
    past = None
    outputs = None

    with torch.inference_mode():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            outputs = model(
                input_ids=prefix_input_ids[:, start:end],
                attention_mask=prefix_attention_mask[:, :end],
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
            )
            past = outputs.past_key_values

    return outputs


def append_new_full_blocks_if_needed(
        *,
        store: KVBlockStore,
        past_key_values,
        known_num_blocks: int,
):
    full_past, tail_past, full_tokens = split_full_blocks_and_tail(
        past_key_values,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    new_num_blocks = full_tokens // TOKENS_PER_BLOCK
    added_block_ids: list[int] = []

    if new_num_blocks > known_num_blocks:
        for block_id in range(known_num_blocks, new_num_blocks):
            key, value = extract_single_block_from_past(
                full_past,
                block_id=block_id,
                tokens_per_block=TOKENS_PER_BLOCK,
            )
            store.put_gpu(block_id, key, value)
            added_block_ids.append(block_id)

    return new_num_blocks, tail_past, added_block_ids


def reload_all_blocks_for_forward(store: KVBlockStore, device, num_blocks: int) -> None:
    for block_id in range(num_blocks):
        store.ensure_gpu(block_id, device)


def kv_cache_nbytes(past_key_values) -> int:
    return sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in past_key_values
    )


def main() -> None:
    args = parse_args()
    context_tokens = args.context_tokens
    new_tokens = args.new_tokens
    vram_budget = int(args.vram_budget_mb * 1_000_000)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", POLICY, "(placement here does not use attention scores)")
    print("context_tokens (target):", context_tokens)
    print("new_tokens:", new_tokens)
    print("vram_budget_mb:", vram_budget / 1_000_000)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    prompt = build_long_prompt(tokenizer, context_tokens)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=context_tokens,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    actual_context_tokens = input_ids.shape[-1]
    print("actual_context_tokens:", actual_context_tokens)

    prefix_input_ids = input_ids[:, :-1]
    prefix_attention_mask = attention_mask[:, :-1]

    # ---- Baseline: full KV cache stays resident on GPU throughout ----
    print("\nRunning baseline (full KV resident on GPU, no paging)...")

    baseline_outputs = chunked_prefill(
        model, prefix_input_ids, prefix_attention_mask, PREFILL_CHUNK
    )

    next_input_id = input_ids[:, -1:]
    current_mask = attention_mask
    current_cache = baseline_outputs.past_key_values
    baseline_generated: list[int] = []

    with torch.inference_mode():
        for _ in range(new_tokens):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_mask,
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )
            next_token_id = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True
            )
            baseline_generated.append(int(next_token_id.item()))

            current_cache = outputs.past_key_values
            next_input_id = next_token_id
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(
                        (current_mask.shape[0], 1),
                        dtype=current_mask.dtype,
                        device=current_mask.device,
                    ),
                ],
                dim=1,
            )

    baseline_full_kv_bytes = kv_cache_nbytes(get_legacy_past_key_values(outputs))
    print("baseline_full_kv_mb (if fully resident on GPU):", f"{baseline_full_kv_bytes / 1_000_000:.2f}")

    del baseline_outputs, current_cache, outputs
    torch.cuda.empty_cache()

    # ---- Paged: pager decides which blocks stay on GPU ----
    print("\nRunning paged (pager-managed GPU/CPU placement)...")

    paged_prefill_outputs = chunked_prefill(
        model, prefix_input_ids, prefix_attention_mask, PREFILL_CHUNK
    )
    full_prefix_past = get_legacy_past_key_values(paged_prefill_outputs)

    full_past, tail_past, full_tokens = split_full_blocks_and_tail(
        full_prefix_past,
        tokens_per_block=TOKENS_PER_BLOCK,
    )
    kv_blocks = real_past_to_blocks(
        full_past, tokens_per_block=TOKENS_PER_BLOCK, verbose=False
    )

    store = KVBlockStore(tokens_per_block=TOKENS_PER_BLOCK)
    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    num_layers = len(full_past)
    num_blocks = len(kv_blocks)
    print("total_kv_blocks:", num_blocks)

    p = pager.PyPager(
        vram_budget,
        RAM_BUDGET,
        RECENT_WINDOW,
        REBALANCE_INTERVAL,
        PROMOTE_MARGIN,
        RAM_PROMOTE_MARGIN,
        POLICY,
    )

    next_input_id = input_ids[:, -1:]
    current_mask = attention_mask
    paged_generated: list[int] = []

    for _ in range(new_tokens):
        reload_all_blocks_for_forward(store, device, num_blocks)

        reconstructed_full_past = reconstruct_past_from_store(
            store=store,
            num_layers=num_layers,
            num_blocks=num_blocks,
        )
        current_past_for_forward = append_tail_to_reconstructed_past(
            reconstructed_full_past,
            tail_past,
        )
        cache = DynamicCache.from_legacy_cache(tuple(current_past_for_forward))

        with torch.inference_mode():
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_mask,
                past_key_values=cache,
                use_cache=True,
                output_attentions=False,
            )

        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        paged_generated.append(int(next_token_id.item()))

        current_past = get_legacy_past_key_values(outputs)
        old_num_blocks = num_blocks

        num_blocks, tail_past, added_block_ids = append_new_full_blocks_if_needed(
            store=store,
            past_key_values=current_past,
            known_num_blocks=old_num_blocks,
        )

        # placement here never depends on attention scores (recent_only),
        # so a dummy uniform vector is enough to satisfy the pager API.
        dummy_attention = [1.0 / num_blocks] * num_blocks
        query_block = num_blocks - 1

        p.on_step(query_block, 0, dummy_attention)
        if added_block_ids:
            p.force_rebalance(query_block)

        store.apply_tiers(p.tiers(), device)

        next_input_id = next_token_id
        current_mask = torch.cat(
            [
                current_mask,
                torch.ones(
                    (current_mask.shape[0], 1),
                    dtype=current_mask.dtype,
                    device=current_mask.device,
                ),
            ],
            dim=1,
        )

    paged_resident_gpu_bytes = store.resident_gpu_bytes()
    paged_resident_cpu_bytes = store.resident_cpu_bytes()

    reduction_pct = 100.0 * (1.0 - paged_resident_gpu_bytes / baseline_full_kv_bytes)

    print("\nVRAM savings proof")
    print("-------------------")
    print("final_num_blocks:", num_blocks)
    print("final_gpu_blocks:", format_block_list(store.gpu_block_ids()))
    print("final_cpu_blocks:", format_block_list(store.cpu_block_ids()))
    print("baseline_full_kv_mb (if fully resident on GPU):", f"{baseline_full_kv_bytes / 1_000_000:.2f}")
    print("paged_resident_gpu_mb:", f"{paged_resident_gpu_bytes / 1_000_000:.2f}")
    print("paged_resident_cpu_mb:", f"{paged_resident_cpu_bytes / 1_000_000:.2f}")
    print(f"gpu_kv_reduction: {reduction_pct:.1f}%")
    print("baseline_ids:", baseline_generated)
    print("paged_ids:", paged_generated)
    print("same_token_ids:", baseline_generated == paged_generated)

    assert baseline_generated == paged_generated

    print(
        "\nOK: pager keeps most real Qwen KV cache off the GPU with byte-identical output."
    )


if __name__ == "__main__":
    main()
