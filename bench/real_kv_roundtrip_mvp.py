from __future__ import annotations

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
    get_legacy_past_key_values,
    real_past_to_blocks,
    reconstruct_past_from_store,
)
from torch_kv_block_store import KVBlockStore
from torch_kv_offload_mvp import print_summary
from transformers import AutoModelForCausalLM, AutoTokenizer

import pager


def compare_past(*, original_past, reconstructed_past, full_tokens: int) -> dict:
    """Compute the largest key/value difference between original and reconstructed KV."""
    max_key_diff = 0.0
    max_value_diff = 0.0

    for (orig_key, orig_value), (rec_key, rec_value) in zip(original_past, reconstructed_past):
        max_key_diff = max(max_key_diff, torch.max(torch.abs(orig_key[:, :, :full_tokens, :] - rec_key)).item())
        max_value_diff = max(max_value_diff, torch.max(torch.abs(orig_value[:, :, :full_tokens, :] - rec_value)).item())

    return {"max_key_diff": max_key_diff, "max_value_diff": max_value_diff}


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

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager", torch_dtype=torch.float16).to(
        device
    )

    model.eval()

    prompt = build_prompt()

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("prompt_seq_len:", input_ids.shape[-1])

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, output_attentions=True)

    original_past = get_legacy_past_key_values(outputs)
    num_layers = len(original_past)

    kv_blocks = real_past_to_blocks(original_past, tokens_per_block=TOKENS_PER_BLOCK)

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
        VRAM_BUDGET, RAM_BUDGET, RECENT_WINDOW, REBALANCE_INTERVAL, PROMOTE_MARGIN, RAM_PROMOTE_MARGIN, POLICY
    )

    block_attention = extract_last_query_block_attention(
        outputs, tokens_per_block=TOKENS_PER_BLOCK, num_blocks=num_blocks
    )

    query_block = num_blocks - 1

    p.on_step(query_block, 0, block_attention)

    movement = store.apply_tiers(p.tiers(), device)

    print_summary("After Rust pager offload placement", store)
    print("moved_to_cpu:", movement["to_cpu"])
    print("pager vram blocks:", p.vram_block_ids())
    print("store gpu blocks:", store.gpu_block_ids())

    # Reconstructing full past requires all blocks back on GPU.
    for block_id in range(num_blocks):
        store.ensure_gpu(block_id, device)

    print_summary("After reloading all blocks for reconstruction", store)

    reconstructed_past = reconstruct_past_from_store(store=store, num_layers=num_layers, num_blocks=num_blocks)

    comparison = compare_past(
        original_past=original_past, reconstructed_past=reconstructed_past, full_tokens=full_tokens
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
