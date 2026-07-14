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
from torch_kv_offload_mvp import bytes_to_mb, print_summary


def compare_logits(
        original_logits: torch.Tensor,
        reconstructed_logits: torch.Tensor,
) -> dict:
    diff = torch.abs(original_logits - reconstructed_logits)

    original_argmax = int(torch.argmax(original_logits, dim=-1).item())
    reconstructed_argmax = int(torch.argmax(reconstructed_logits, dim=-1).item())

    return {
        "max_logits_diff": float(torch.max(diff).item()),
        "mean_logits_diff": float(torch.mean(diff).item()),
        "original_argmax": original_argmax,
        "reconstructed_argmax": reconstructed_argmax,
        "same_argmax": original_argmax == reconstructed_argmax,
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

    kv_blocks = real_past_to_blocks(
        original_past,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    num_blocks = len(kv_blocks)
    num_layers = len(original_past)
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

    # Reload all blocks to reconstruct full KV.
    for block_id in range(num_blocks):
        store.ensure_gpu(block_id, device)

    reconstructed_past = reconstruct_past_from_store(
        store=store,
        num_layers=num_layers,
        num_blocks=num_blocks,
    )

    print_summary("After reloading all blocks for logits test", store)

    # We only reconstructed full blocks.
    # So compare logits using the next token after the reconstructed prefix.
    prefix_input_ids = input_ids[:, :full_tokens]
    prefix_attention_mask = attention_mask[:, :full_tokens]

    next_input_id = input_ids[:, full_tokens:full_tokens + 1]

    if next_input_id.shape[-1] != 1:
        raise RuntimeError("No next token available after reconstructed prefix.")

    # Original full-block past from original_past.
    original_trimmed_past = []

    for key, value in original_past:
        original_trimmed_past.append(
            (
                key[:, :, :full_tokens, :].contiguous(),
                value[:, :, :full_tokens, :].contiguous(),
            )
        )

    # Attention mask for prefix + one next token.
    next_attention_mask = attention_mask[:, :full_tokens + 1]

    original_cache = DynamicCache.from_legacy_cache(
        tuple(original_trimmed_past)
    )
    reconstructed_cache = DynamicCache.from_legacy_cache(
        tuple(reconstructed_past)
    )

    with torch.inference_mode():
        original_next = model(
            input_ids=next_input_id,
            attention_mask=next_attention_mask,
            past_key_values=original_cache,
            use_cache=False,
        )

        reconstructed_next = model(
            input_ids=next_input_id,
            attention_mask=next_attention_mask,
            past_key_values=reconstructed_cache,
            use_cache=False,
        )

    original_logits = original_next.logits[:, -1, :]
    reconstructed_logits = reconstructed_next.logits[:, -1, :]

    comparison = compare_logits(
        original_logits=original_logits,
        reconstructed_logits=reconstructed_logits,
    )

    original_token = tokenizer.decode([comparison["original_argmax"]])
    reconstructed_token = tokenizer.decode([comparison["reconstructed_argmax"]])

    print("\nLogits roundtrip comparison")
    print("---------------------------")
    print("max_logits_diff:", comparison["max_logits_diff"])
    print("mean_logits_diff:", comparison["mean_logits_diff"])
    print("same_argmax:", comparison["same_argmax"])
    print("original_argmax:", comparison["original_argmax"], repr(original_token))
    print(
        "reconstructed_argmax:",
        comparison["reconstructed_argmax"],
        repr(reconstructed_token),
    )

    assert comparison["max_logits_diff"] == 0.0
    assert comparison["mean_logits_diff"] == 0.0
    assert comparison["same_argmax"] is True

    print("\nOK: reconstructed real Qwen KV cache produces identical next-token logits.")


if __name__ == "__main__":
    main()
