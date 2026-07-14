from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

import pager
from real_kv_cache_block_mvp import (
    MODEL_NAME,
    TOKENS_PER_BLOCK,
    MAX_LENGTH,
    VRAM_BUDGET,
    RAM_BUDGET,
    RECENT_WINDOW,
    REBALANCE_INTERVAL,
    PROMOTE_MARGIN,
    RAM_PROMOTE_MARGIN,
    POLICY,
    build_prompt,
    get_legacy_past_key_values,
    real_past_to_blocks,
    extract_last_query_block_attention,
)
from real_kv_roundtrip_mvp import reconstruct_past_from_store
from torch_kv_block_store import KVBlockStore
from torch_kv_offload_mvp import print_summary


GENERATE_TOKENS = 8


def format_block_list(block_ids: list[int], limit: int = 30) -> str:
    if len(block_ids) <= limit:
        return str(block_ids)

    shown = block_ids[:limit]
    remaining = len(block_ids) - limit
    return f"{shown} ... (+{remaining} more)"


def trim_tail_to_full_blocks(past_key_values):
    """
    Keep only full TOKENS_PER_BLOCK blocks.
    """
    seq_len = past_key_values[0][0].shape[2]
    full_tokens = (seq_len // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK

    trimmed = []

    for key, value in past_key_values:
        trimmed.append(
            (
                key[:, :, :full_tokens, :].contiguous(),
                value[:, :, :full_tokens, :].contiguous(),
            )
        )

    return trimmed, full_tokens

def split_full_blocks_and_tail(past_key_values):
    seq_len = past_key_values[0][0].shape[2]
    full_tokens = (seq_len // TOKENS_PER_BLOCK) * TOKENS_PER_BLOCK

    full_past = []
    tail_past = []

    for key, value in past_key_values:
        full_past.append(
            (
                key[:, :, :full_tokens, :].contiguous(),
                value[:, :, :full_tokens, :].contiguous(),
            )
        )
        tail_past.append(
            (
                key[:, :, full_tokens:, :].contiguous(),
                value[:, :, full_tokens:, :].contiguous(),
            )
        )

    return full_past, tail_past, full_tokens


def append_tail_to_reconstructed_past(
        reconstructed_past,
        tail_past,
):
    out = []

    for (rec_key, rec_value), (tail_key, tail_value) in zip(
            reconstructed_past,
            tail_past,
    ):
        key = torch.cat([rec_key, tail_key], dim=2).contiguous()
        value = torch.cat([rec_value, tail_value], dim=2).contiguous()
        out.append((key, value))

    return out

def rebuild_store_from_past(
        past_key_values,
) -> tuple[KVBlockStore, int, int, list[tuple[torch.Tensor, torch.Tensor]], int]:
    """
    Convert only full KV blocks into KVBlockStore.

    Tail tokens that do not fill a complete block are returned separately
    and appended back before the next model forward.
    """
    full_past, tail_past, full_tokens = split_full_blocks_and_tail(
        past_key_values
    )

    kv_blocks = real_past_to_blocks(
        full_past,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    store = KVBlockStore(tokens_per_block=TOKENS_PER_BLOCK)

    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    num_layers = len(full_past)
    num_blocks = len(kv_blocks)

    return store, num_layers, num_blocks, tail_past, full_tokens


def apply_pager_to_store(
        *,
        p: pager.PyPager,
        store: KVBlockStore,
        block_attention: list[float],
        query_block: int,
        device: torch.device,
) -> dict[str, list[int]]:
    p.on_step(query_block, 0, block_attention)
    tiers = p.tiers()
    return store.apply_tiers(tiers, device)


def reload_all_blocks(
        *,
        store: KVBlockStore,
        num_blocks: int,
        device: torch.device,
) -> None:
    for block_id in range(num_blocks):
        store.ensure_gpu(block_id, device)


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

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    ).to(device)

    model.eval()

    prompt = build_prompt()

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("prompt_seq_len:", input_ids.shape[-1])

    # Build cache for prompt[:-1], then feed prompt[-1] as the first decode input.
    # This makes step 1 generate the first token after the full prompt.
    prefix_input_ids = input_ids[:, :-1]
    prefix_attention_mask = attention_mask[:, :-1]

    next_input_id = input_ids[:, -1:]
    current_attention_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=prefix_input_ids,
            attention_mask=prefix_attention_mask,
            use_cache=True,
            output_attentions=True,
        )

    current_past = get_legacy_past_key_values(outputs)

    print("initial_cache_tokens:", prefix_input_ids.shape[-1])
    print(
        "initial_next_input_id:",
        int(next_input_id.item()),
        repr(tokenizer.decode(next_input_id[0])),
    )

    p = pager.PyPager(
        VRAM_BUDGET,
        RAM_BUDGET,
        RECENT_WINDOW,
        REBALANCE_INTERVAL,
        PROMOTE_MARGIN,
        RAM_PROMOTE_MARGIN,
        POLICY,
    )

    generated: list[int] = []

    total_gpu_to_cpu_mb = 0.0
    total_cpu_to_gpu_mb = 0.0
    total_gpu_to_cpu_copies = 0
    total_cpu_to_gpu_copies = 0

    for step in range(1, GENERATE_TOKENS + 1):
        store, num_layers, num_blocks, tail_past, full_tokens = rebuild_store_from_past(
            current_past
        )

        tail_tokens = tail_past[0][0].shape[2]

        block_attention = extract_last_query_block_attention(
            outputs,
            tokens_per_block=TOKENS_PER_BLOCK,
            num_blocks=num_blocks,
        )

        query_block = num_blocks - 1

        movement = apply_pager_to_store(
            p=p,
            store=store,
            block_attention=block_attention,
            query_block=query_block,
            device=device,
        )

        summary_after_offload = store.summary()

        gpu_to_cpu_before_reload = summary_after_offload["gpu_to_cpu_bytes"]
        cpu_to_gpu_before_reload = summary_after_offload["cpu_to_gpu_bytes"]
        gpu_to_cpu_copies_before_reload = summary_after_offload["gpu_to_cpu_copies"]
        cpu_to_gpu_copies_before_reload = summary_after_offload["cpu_to_gpu_copies"]

        real_attention_in_gpu = sum(
            block_attention[block_id]
            for block_id in store.gpu_block_ids()
            if block_id < len(block_attention)
        )

        print(f"\nPaged generation step {step}")
        print("-----------------------")
        print("num_blocks:", num_blocks)
        print("query_block:", query_block)
        print("full_tokens:", full_tokens)
        print("tail_tokens:", tail_tokens)
        print("moved_to_cpu:", format_block_list(movement["to_cpu"]))
        print("moved_to_gpu:", format_block_list(movement["to_gpu"]))
        print("pager_vram_blocks:", format_block_list(p.vram_block_ids()))
        print("store_gpu_blocks:", format_block_list(store.gpu_block_ids()))
        print("real_attention_in_gpu:", f"{real_attention_in_gpu:.4f}")
        print(
            "resident_gpu_mb_after_offload:",
            f"{summary_after_offload['resident_gpu_bytes'] / 1_000_000:.2f}",
        )
        print(
            "resident_cpu_mb_after_offload:",
            f"{summary_after_offload['resident_cpu_bytes'] / 1_000_000:.2f}",
        )

        # HF forward still requires full GPU KV, so reload all blocks before reconstructing.
        reload_all_blocks(
            store=store,
            num_blocks=num_blocks,
            device=device,
        )

        summary_after_reload = store.summary()

        total_gpu_to_cpu_mb += summary_after_reload["gpu_to_cpu_bytes"] / 1_000_000
        total_cpu_to_gpu_mb += summary_after_reload["cpu_to_gpu_bytes"] / 1_000_000
        total_gpu_to_cpu_copies += summary_after_reload["gpu_to_cpu_copies"]
        total_cpu_to_gpu_copies += summary_after_reload["cpu_to_gpu_copies"]

        step_cpu_to_gpu_mb = (
                                     summary_after_reload["cpu_to_gpu_bytes"] - cpu_to_gpu_before_reload
                             ) / 1_000_000
        step_cpu_to_gpu_copies = (
                summary_after_reload["cpu_to_gpu_copies"] - cpu_to_gpu_copies_before_reload
        )
        print("cpu_to_gpu_reload_mb:", f"{step_cpu_to_gpu_mb:.2f}")
        print("cpu_to_gpu_reload_copies:", step_cpu_to_gpu_copies)

        reconstructed_full_past = reconstruct_past_from_store(
            store=store,
            num_layers=num_layers,
            num_blocks=num_blocks,
        )

        current_past = append_tail_to_reconstructed_past(
            reconstructed_full_past,
            tail_past,
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

        logits = outputs.logits[:, -1, :]
        generated_token = torch.argmax(logits, dim=-1, keepdim=True)

        token_id = int(generated_token.item())
        generated.append(token_id)

        print("generated_token_id:", token_id)
        print("generated_token:", repr(tokenizer.decode([token_id])))

        current_past = get_legacy_past_key_values(outputs)

        next_input_id = generated_token
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

    generated_text = tokenizer.decode(generated)

    print("\nPaged generation loop summary")
    print("-----------------------------")
    print("generated_ids:", generated)
    print("generated_text:", repr(generated_text))
    print("total_gpu_to_cpu_mb:", f"{total_gpu_to_cpu_mb:.2f}")
    print("total_cpu_to_gpu_mb:", f"{total_cpu_to_gpu_mb:.2f}")
    print("total_gpu_to_cpu_copies:", total_gpu_to_cpu_copies)
    print("total_cpu_to_gpu_copies:", total_cpu_to_gpu_copies)

    metrics = p.metrics()

    print("\nPager metrics")
    print("-------------")
    print("tokens:", metrics.tokens)
    print("vram_peak_mb:", metrics.vram_peak / 1_000_000)
    print("ram_peak_mb:", metrics.ram_peak / 1_000_000)
    print("swap_vram_ram_mb:", metrics.swap_vram_ram / 1_000_000)
    print("swap_ram_ssd_mb:", metrics.swap_ram_ssd / 1_000_000)

    print("\nOK: real Qwen KV cache was paged in a multi-step generation loop.")


if __name__ == "__main__":
    main()
