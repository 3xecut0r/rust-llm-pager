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
    format_block_list,
    get_legacy_past_key_values,
    real_past_to_blocks,
)
from torch_kv_block_store import KVBlockStore
from torch_kv_offload_mvp import bytes_to_mb, print_summary
from transformers import AutoModelForCausalLM, AutoTokenizer

import pager


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", POLICY)
    print("tokens_per_block:", TOKENS_PER_BLOCK)
    print("max_length:", MAX_LENGTH)
    print("pager_vram_budget_mb:", bytes_to_mb(VRAM_BUDGET))
    print("pager_ram_budget_mb:", bytes_to_mb(RAM_BUDGET))

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

    past_key_values = get_legacy_past_key_values(outputs)

    kv_blocks = real_past_to_blocks(past_key_values, tokens_per_block=TOKENS_PER_BLOCK)

    if not kv_blocks:
        raise RuntimeError("No full KV blocks were extracted. Increase prompt length or reduce TOKENS_PER_BLOCK.")

    store = KVBlockStore(tokens_per_block=TOKENS_PER_BLOCK)

    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    print("real_model_kv_blocks:", len(kv_blocks))
    print("real_model_kv_block_size_mb:", f"{store.resident_gpu_bytes() / len(kv_blocks) / 1_000_000:.3f}")

    print_summary("After real model KV extraction", store)

    p = pager.PyPager(
        VRAM_BUDGET, RAM_BUDGET, RECENT_WINDOW, REBALANCE_INTERVAL, PROMOTE_MARGIN, RAM_PROMOTE_MARGIN, POLICY
    )

    block_attention = extract_last_query_block_attention(
        outputs, tokens_per_block=TOKENS_PER_BLOCK, num_blocks=len(kv_blocks)
    )

    query_block = len(kv_blocks) - 1

    # In this MVP, pager logical blocks are aligned with extracted KV block ids.
    p.on_step(query_block, 0, block_attention)

    movement = store.apply_tiers(p.tiers(), device)

    print_summary("After Rust pager placement on real model KV", store)

    print("moved_to_gpu:", format_block_list(movement["to_gpu"]))
    print("moved_to_cpu:", format_block_list(movement["to_cpu"]))
    print("pager vram blocks:", p.vram_block_ids())
    print("store gpu blocks:", store.gpu_block_ids())
    real_attention_in_gpu = sum(
        block_attention[block_id] for block_id in store.gpu_block_ids() if block_id < len(block_attention)
    )

    print("real_attention_in_gpu:", f"{real_attention_in_gpu:.4f}")

    metrics = p.metrics()

    print("\nPager metrics")
    print("-------------")
    print("tokens:", metrics.tokens)
    print("vram_peak_mb:", bytes_to_mb(metrics.vram_peak))
    print("ram_peak_mb:", bytes_to_mb(metrics.ram_peak))
    print("swap_vram_ram_mb:", bytes_to_mb(metrics.swap_vram_ram))
    print("swap_ram_ssd_mb:", bytes_to_mb(metrics.swap_ram_ssd))
    print("attention_mass_total:", f"{metrics.attention_mass_total:.4f}")
    print(
        "note:",
        "pager internal attention_mass_vram is not used here because this script "
        "measures real_attention_in_gpu after physical placement.",
    )

    assert p.vram_block_ids() == store.gpu_block_ids()

    print("\nOK: real Qwen past_key_values were split into blocks and placed by Rust pager.")


if __name__ == "__main__":
    main()
