from __future__ import annotations

import torch
from config import (
    GENERATE_TOKENS,
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


def greedy_baseline_generate(
    *, model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int
) -> list[int]:
    """Generate steps tokens the plain way, with the full KV cache always resident on GPU."""
    next_input_id = input_ids[:, -1:]
    current_attention_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True, output_attentions=False
        )

    current_past = outputs.past_key_values
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_attention_mask,
                past_key_values=current_past,
                use_cache=True,
                output_attentions=False,
            )

            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(next_token_id.item()))

            current_past = outputs.past_key_values
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


def rebuild_store_from_full_past(full_past) -> tuple[KVBlockStore, int, int]:
    """Build a fresh KVBlockStore from a full-blocks-only past_key_values."""
    kv_blocks = real_past_to_blocks(full_past, tokens_per_block=TOKENS_PER_BLOCK)

    store = KVBlockStore(tokens_per_block=TOKENS_PER_BLOCK)

    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    return store, len(full_past), len(kv_blocks)


def paged_generate(
    *, model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int
) -> tuple[list[int], dict]:
    """Run the paged generation loop and return the generated tokens plus swap stats."""
    next_input_id = input_ids[:, -1:]
    current_attention_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True, output_attentions=True
        )

    current_past = get_legacy_past_key_values(outputs)

    p = pager.PyPager(
        VRAM_BUDGET, RAM_BUDGET, RECENT_WINDOW, REBALANCE_INTERVAL, PROMOTE_MARGIN, RAM_PROMOTE_MARGIN, POLICY
    )

    generated: list[int] = []

    total_gpu_to_cpu_bytes = 0
    total_cpu_to_gpu_bytes = 0
    total_gpu_to_cpu_copies = 0
    total_cpu_to_gpu_copies = 0
    attention_in_gpu_values: list[float] = []

    for step in range(1, steps + 1):
        full_past, tail_past, full_tokens = split_full_blocks_and_tail(current_past, tokens_per_block=TOKENS_PER_BLOCK)

        store, num_layers, num_blocks = rebuild_store_from_full_past(full_past)

        block_attention = extract_last_query_block_attention(
            outputs, tokens_per_block=TOKENS_PER_BLOCK, num_blocks=num_blocks
        )

        query_block = num_blocks - 1

        p.on_step(query_block, 0, block_attention)

        movement = store.apply_tiers(p.tiers(), input_ids.device)

        summary_after_offload = store.summary()

        real_attention_in_gpu = sum(
            block_attention[block_id] for block_id in store.gpu_block_ids() if block_id < len(block_attention)
        )
        attention_in_gpu_values.append(real_attention_in_gpu)

        print(f"\nPaged compare step {step}")
        print("--------------------")
        print("full_tokens:", full_tokens)
        print("tail_tokens:", tail_past[0][0].shape[2])
        print("num_blocks:", num_blocks)
        print("moved_to_cpu:", format_block_list(movement["to_cpu"]))
        print("pager_vram_blocks:", format_block_list(p.vram_block_ids()))
        print("store_gpu_blocks:", format_block_list(store.gpu_block_ids()))
        print("real_attention_in_gpu:", f"{real_attention_in_gpu:.4f}")
        print("resident_gpu_mb_after_offload:", f"{summary_after_offload['resident_gpu_bytes'] / 1_000_000:.2f}")
        print("resident_cpu_mb_after_offload:", f"{summary_after_offload['resident_cpu_bytes'] / 1_000_000:.2f}")

        for block_id in range(num_blocks):
            store.ensure_gpu(block_id, input_ids.device)

        summary_after_reload = store.summary()

        total_gpu_to_cpu_bytes += summary_after_reload["gpu_to_cpu_bytes"]
        total_cpu_to_gpu_bytes += summary_after_reload["cpu_to_gpu_bytes"]
        total_gpu_to_cpu_copies += summary_after_reload["gpu_to_cpu_copies"]
        total_cpu_to_gpu_copies += summary_after_reload["cpu_to_gpu_copies"]

        reload_bytes = summary_after_reload["cpu_to_gpu_bytes"] - summary_after_offload["cpu_to_gpu_bytes"]
        print("cpu_to_gpu_reload_mb:", f"{reload_bytes / 1_000_000:.2f}")
        print(
            "cpu_to_gpu_reload_copies:",
            summary_after_reload["cpu_to_gpu_copies"] - summary_after_offload["cpu_to_gpu_copies"],
        )

        current_past = reconstruct_past_from_store(
            store=store, num_layers=num_layers, num_blocks=num_blocks, tail_past=tail_past
        )

        cache = DynamicCache.from_legacy_cache(tuple(current_past))

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

        print("generated_token_id:", token_id)
        print("generated_token:", repr(tokenizer.decode([token_id])))

        current_past = get_legacy_past_key_values(outputs)
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

    stats = {
        "total_gpu_to_cpu_mb": total_gpu_to_cpu_bytes / 1_000_000,
        "total_cpu_to_gpu_mb": total_cpu_to_gpu_bytes / 1_000_000,
        "total_gpu_to_cpu_copies": total_gpu_to_cpu_copies,
        "total_cpu_to_gpu_copies": total_cpu_to_gpu_copies,
        "mean_attention_in_gpu": (
            sum(attention_in_gpu_values) / len(attention_in_gpu_values) if attention_in_gpu_values else 0.0
        ),
        "min_attention_in_gpu": (min(attention_in_gpu_values) if attention_in_gpu_values else 0.0),
        "max_attention_in_gpu": (max(attention_in_gpu_values) if attention_in_gpu_values else 0.0),
    }

    return generated, stats


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", POLICY)
    print("tokens_per_block:", TOKENS_PER_BLOCK)
    print("generate_tokens:", GENERATE_TOKENS)

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
        model=model, tokenizer=tokenizer, input_ids=input_ids, attention_mask=attention_mask, steps=GENERATE_TOKENS
    )

    paged_generated, paged_stats = paged_generate(
        model=model, tokenizer=tokenizer, input_ids=input_ids, attention_mask=attention_mask, steps=GENERATE_TOKENS
    )

    print("\nBaseline vs paged generation comparison")
    print("---------------------------------------")
    print("baseline_ids:", baseline_generated)
    print("paged_ids:", paged_generated)
    print("same_token_ids:", baseline_generated == paged_generated)
    print("baseline_text:", repr(tokenizer.decode(baseline_generated)))
    print("paged_text:", repr(tokenizer.decode(paged_generated)))

    print("\nPaged stats")
    print("-----------")
    print("total_gpu_to_cpu_mb:", f"{paged_stats['total_gpu_to_cpu_mb']:.2f}")
    print("total_cpu_to_gpu_mb:", f"{paged_stats['total_cpu_to_gpu_mb']:.2f}")
    print("total_gpu_to_cpu_copies:", paged_stats["total_gpu_to_cpu_copies"])
    print("total_cpu_to_gpu_copies:", paged_stats["total_cpu_to_gpu_copies"])
    print("mean_attention_in_gpu:", f"{paged_stats['mean_attention_in_gpu']:.4f}")
    print("min_attention_in_gpu:", f"{paged_stats['min_attention_in_gpu']:.4f}")
    print("max_attention_in_gpu:", f"{paged_stats['max_attention_in_gpu']:.4f}")

    assert baseline_generated == paged_generated

    print("\nOK: paged real Qwen KV loop produces identical greedy generation " "to baseline.")


if __name__ == "__main__":
    main()
