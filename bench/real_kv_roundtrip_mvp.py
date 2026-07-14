from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from torch_kv_block_store import KVBlockStore
from torch_kv_offload_mvp import bytes_to_mb, print_summary

import pager


def reconstruct_past_from_store(
        *,
        store: KVBlockStore,
        num_layers: int,
        num_blocks: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Reconstruct legacy past_key_values from GPU-resident KV blocks.

    Input block shape:
        [layers, tokens_per_block, kv_heads, head_dim]

    Output per layer:
        key/value shape: [batch, kv_heads, seq_len, head_dim]
    """
    layer_keys: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    layer_values: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

    for block_id in range(num_blocks):
        block_key, block_value = store.get_gpu(block_id)

        for layer_idx in range(num_layers):
            key_slice = block_key[layer_idx]      # [tokens, heads, dim]
            value_slice = block_value[layer_idx]  # [tokens, heads, dim]

            # [tokens, heads, dim] -> [heads, tokens, dim]
            layer_keys[layer_idx].append(
                key_slice.permute(1, 0, 2).contiguous()
            )
            layer_values[layer_idx].append(
                value_slice.permute(1, 0, 2).contiguous()
            )

    reconstructed = []

    for layer_idx in range(num_layers):
        # [heads, seq, dim]
        key = torch.cat(layer_keys[layer_idx], dim=1)
        value = torch.cat(layer_values[layer_idx], dim=1)

        # [batch, heads, seq, dim]
        key = key.unsqueeze(0).contiguous()
        value = value.unsqueeze(0).contiguous()

        reconstructed.append((key, value))

    return reconstructed


def compare_past(
        *,
        original_past,
        reconstructed_past,
        full_tokens: int,
) -> dict:
    max_key_diff = 0.0
    max_value_diff = 0.0

    for (orig_key, orig_value), (rec_key, rec_value) in zip(
            original_past,
            reconstructed_past,
    ):
        orig_key = orig_key[:, :, :full_tokens, :]
        orig_value = orig_value[:, :, :full_tokens, :]

        key_diff = torch.max(torch.abs(orig_key - rec_key)).item()
        value_diff = torch.max(torch.abs(orig_value - rec_value)).item()

        max_key_diff = max(max_key_diff, key_diff)
        max_value_diff = max(max_value_diff, value_diff)

    return {
        "max_key_diff": max_key_diff,
        "max_value_diff": max_value_diff,
    }


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", POLICY)
    print("tokens_per_block:", TOKENS_PER_BLOCK)

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

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            output_attentions=True,
        )

    original_past = get_legacy_past_key_values(outputs)
    num_layers = len(original_past)

    kv_blocks = real_past_to_blocks(
        original_past,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    num_blocks = len(kv_blocks)
    full_tokens = num_blocks * TOKENS_PER_BLOCK

    print("num_layers:", num_layers)
    print("num_blocks:", num_blocks)
    print("full_tokens:", full_tokens)
    print("ignored_tail_tokens:", input_ids.shape[-1] - full_tokens)

    store = KVBlockStore(tokens_per_block=TOKENS_PER_BLOCK)

    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    print_summary("After real KV block extraction", store)

    p = pager.PyPager(
        VRAM_BUDGET,
        RAM_BUDGET,
        RECENT_WINDOW,
        REBALANCE_INTERVAL,
        PROMOTE_MARGIN,
        RAM_PROMOTE_MARGIN,
        POLICY,
    )

    block_attention = extract_last_query_block_attention(
        outputs,
        tokens_per_block=TOKENS_PER_BLOCK,
        num_blocks=num_blocks,
    )

    query_block = num_blocks - 1

    p.on_step(query_block, 0, block_attention)

    tiers = p.tiers()
    movement = store.apply_tiers(tiers, device)

    print_summary("After Rust pager offload placement", store)
    print("moved_to_cpu:", movement["to_cpu"])
    print("pager vram blocks:", p.vram_block_ids())
    print("store gpu blocks:", store.gpu_block_ids())

    # Reconstructing full past requires all blocks back on GPU.
    for block_id in range(num_blocks):
        store.ensure_gpu(block_id, device)

    print_summary("After reloading all blocks for reconstruction", store)

    reconstructed_past = reconstruct_past_from_store(
        store=store,
        num_layers=num_layers,
        num_blocks=num_blocks,
    )

    comparison = compare_past(
        original_past=original_past,
        reconstructed_past=reconstructed_past,
        full_tokens=full_tokens,
    )

    print("\nRoundtrip comparison")
    print("--------------------")
    print("max_key_diff:", comparison["max_key_diff"])
    print("max_value_diff:", comparison["max_value_diff"])

    assert comparison["max_key_diff"] == 0.0
    assert comparison["max_value_diff"] == 0.0

    print("\nOK: real Qwen KV blocks survived GPU -> CPU -> GPU roundtrip exactly.")


if __name__ == "__main__":
    main()
