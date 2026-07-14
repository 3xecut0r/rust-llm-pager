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


def trim_past_to_full_blocks(
        past_key_values,
        *,
        full_tokens: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    trimmed = []

    for key, value in past_key_values:
        trimmed.append(
            (
                key[:, :, :full_tokens, :].contiguous(),
                value[:, :, :full_tokens, :].contiguous(),
            )
        )

    return trimmed


def greedy_decode_from_cache(
        *,
        model,
        tokenizer,
        past_key_values,
        next_input_id: torch.Tensor,
        base_attention_mask: torch.Tensor,
        steps: int,
) -> list[int]:
    """
    Greedy decode from an existing KV cache.

    past_key_values:
        legacy list/tuple of per-layer (key, value)
        each key/value shape: [batch, kv_heads, seq_len, head_dim]

    next_input_id:
        first token after cached prefix, shape [1, 1]

    base_attention_mask:
        attention mask for cached prefix + next_input_id, shape [1, prefix_len + 1]
    """
    cache = DynamicCache.from_legacy_cache(tuple(past_key_values))

    current_input_id = next_input_id
    attention_mask = base_attention_mask

    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(
                input_ids=current_input_id,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
            )

            logits = outputs.logits[:, -1, :]
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            token_id = int(next_token_id.item())
            generated.append(token_id)

            cache = outputs.past_key_values
            current_input_id = next_token_id

            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    ),
                ],
                dim=1,
            )

    return generated


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

    next_input_id = input_ids[:, full_tokens:full_tokens + 1]

    if next_input_id.shape[-1] != 1:
        raise RuntimeError("No next token available after reconstructed prefix.")

    base_attention_mask = attention_mask[:, :full_tokens + 1]

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

    # Reload all blocks before reconstructing the full prefix KV cache.
    for block_id in range(num_blocks):
        store.ensure_gpu(block_id, device)

    reconstructed_past = reconstruct_past_from_store(
        store=store,
        num_layers=num_layers,
        num_blocks=num_blocks,
    )

    print_summary("After reloading all blocks for generation test", store)

    original_trimmed_past = trim_past_to_full_blocks(
        original_past,
        full_tokens=full_tokens,
    )

    original_generated = greedy_decode_from_cache(
        model=model,
        tokenizer=tokenizer,
        past_key_values=original_trimmed_past,
        next_input_id=next_input_id,
        base_attention_mask=base_attention_mask,
        steps=GENERATE_TOKENS,
    )

    reconstructed_generated = greedy_decode_from_cache(
        model=model,
        tokenizer=tokenizer,
        past_key_values=reconstructed_past,
        next_input_id=next_input_id,
        base_attention_mask=base_attention_mask,
        steps=GENERATE_TOKENS,
    )

    original_text = tokenizer.decode(original_generated)
    reconstructed_text = tokenizer.decode(reconstructed_generated)

    print("\nGeneration roundtrip comparison")
    print("-------------------------------")
    print("original_generated_ids:", original_generated)
    print("reconstructed_generated_ids:", reconstructed_generated)
    print("same_token_ids:", original_generated == reconstructed_generated)
    print("original_text:", repr(original_text))
    print("reconstructed_text:", repr(reconstructed_text))

    assert original_generated == reconstructed_generated

    print(
        "\nOK: reconstructed real Qwen KV cache produces identical "
        "multi-token greedy generation."
    )


if __name__ == "__main__":
    main()
