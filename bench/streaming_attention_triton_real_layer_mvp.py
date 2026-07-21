from __future__ import annotations

import argparse
import time
import types

import torch
from config import MODEL_NAME, RAM_BUDGET, RECENT_WINDOW, TOKENS_PER_BLOCK, VRAM_BUDGET
from real_kv_utils import build_prompt, get_legacy_past_key_values, real_past_to_blocks, split_full_blocks_and_tail
from real_kv_vram_savings_proof_mvp import chunked_prefill
from streaming_attention_triton_kernel_mvp import triton_decode_attention
from torch_kv_block_store import KVBlockStore
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from pager_hf import PagedModel

STREAM_CHUNK_TOKENS = TOKENS_PER_BLOCK
LAYER_TO_PATCH = 0
NEW_TOKENS_FOR_BASELINE_TIMING = 8
PREFILL_CHUNK = 512


def parse_args() -> argparse.Namespace:
    """Parse --context-tokens, to test both a short and a long (hundreds-of-blocks) context."""
    parser = argparse.ArgumentParser(description="Stage C: Triton kernel hooked into one real Qwen2Attention layer.")
    parser.add_argument("--context-tokens", type=int, default=None, help="pad/repeat the prompt to roughly this length")
    return parser.parse_args()


def gather_layer_kv(
    *, store: KVBlockStore, layer_idx: int, num_blocks: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Gather one layer's historical blocks into one contiguous [kv_heads, total_tokens, head_dim]
    K and V tensor, writing each block directly into a pre-allocated destination (same pattern as
    reconstruct_past_from_store) instead of collecting copies in a list and torch.cat-ing them.
    """
    store.ensure_gpu(0, device)
    sample_key, _ = store.get_gpu(0)
    block_len, kv_heads, head_dim = sample_key.shape[1], sample_key.shape[2], sample_key.shape[3]
    total_tokens = num_blocks * block_len

    dest_key = torch.empty((kv_heads, total_tokens, head_dim), dtype=sample_key.dtype, device=device)
    dest_value = torch.empty_like(dest_key)

    for block_id in range(num_blocks):
        store.ensure_gpu(block_id, device)
        block_key, block_value = store.get_gpu(block_id)
        start = block_id * block_len
        end = start + block_len
        dest_key[:, start:end, :] = block_key[layer_idx].permute(1, 0, 2)
        dest_value[:, start:end, :] = block_value[layer_idx].permute(1, 0, 2)

    return dest_key, dest_value


def build_triton_streaming_forward(
    store: KVBlockStore, layer_idx: int, device: torch.device, num_blocks: int, tail_past, timings: list
):
    """Same contract as the pure-Python version's forward replacement, but the attention step is one Triton kernel launch."""

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
    ):
        bsz, q_len, _ = hidden_states.size()
        assert bsz == 1 and q_len == 1, "this prototype only handles single-token batch=1 decode steps"

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        kv_seq_len = key_states.shape[-2] + past_key_value.get_usable_length(key_states.shape[-2], layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

        torch.cuda.synchronize()
        started = time.perf_counter()

        hist_key, hist_value = gather_layer_kv(store=store, layer_idx=layer_idx, num_blocks=num_blocks, device=device)
        # tail_past: leftover prefix tokens that didn't fill a full block, never went into the store at all --
        # still part of the real context, so they must still be attended to.
        tail_key, tail_value = tail_past[layer_idx][0][0], tail_past[layer_idx][1][0]
        full_key = torch.cat([hist_key, tail_key, key_states[0]], dim=1)
        full_value = torch.cat([hist_value, tail_value, value_states[0]], dim=1)

        q = query_states[0, :, 0, :]
        streaming_out = triton_decode_attention(q, full_key, full_value)

        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)

        attn_output = streaming_out.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward


def build_long_prompt(tokenizer, target_tokens: int) -> str:
    """Repeat a filler paragraph until it tokenizes to at least target_tokens."""
    paragraph = (
        "The quick brown fox jumps over the lazy dog while researchers discuss "
        "memory management, operating systems, distributed caches, and GPU "
        "scheduling in long, unrelated technical documents. "
    )
    paragraph_tokens = len(tokenizer(paragraph)["input_ids"])
    repeats = target_tokens // max(paragraph_tokens, 1) + 4
    return paragraph * repeats


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")
    print("device:", device)
    print("model:", MODEL_NAME)
    print("layer_patched:", LAYER_TO_PATCH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()

    prompt = build_long_prompt(tokenizer, args.context_tokens) if args.context_tokens else build_prompt()
    max_length = args.context_tokens if args.context_tokens else None
    encoded = tokenizer(prompt, return_tensors="pt", truncation=bool(max_length), max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("prompt_seq_len:", input_ids.shape[-1])

    prefill_outputs = chunked_prefill(model, input_ids[:, :-1], attention_mask[:, :-1], PREFILL_CHUNK)
    full_prefix_past = get_legacy_past_key_values(prefill_outputs)

    full_past, tail_past, full_tokens = split_full_blocks_and_tail(
        full_prefix_past, tokens_per_block=STREAM_CHUNK_TOKENS
    )
    kv_blocks = real_past_to_blocks(full_past, tokens_per_block=STREAM_CHUNK_TOKENS, verbose=False)
    num_blocks = len(kv_blocks)
    print("full_tokens:", full_tokens, "tail_tokens:", tail_past[0][0].shape[2], "num_blocks:", num_blocks)

    # Real GPU<->CPU split: most blocks start on GPU then get pushed to CPU,
    # exactly like the pager would between steps, so the gather step has to
    # genuinely stream some blocks back from CPU.
    store = KVBlockStore(tokens_per_block=STREAM_CHUNK_TOKENS)
    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)
    for block_id in range(max(num_blocks - 2, 0)):
        store.offload_to_cpu(block_id)
    print("gpu_resident_blocks (of", num_blocks, "):", len(store.gpu_block_ids()))
    print("cpu_resident_blocks (of", num_blocks, "):", len(store.cpu_block_ids()))

    next_input_id = input_ids[:, -1:]
    next_position = torch.tensor([[input_ids.shape[-1] - 1]], device=device)
    next_cache_position = torch.tensor([input_ids.shape[-1] - 1], device=device)

    print("\nRunning stock forward (unpatched, full reconstructed cache)...")
    with torch.inference_mode():
        stock_outputs = model(
            input_ids=next_input_id,
            attention_mask=attention_mask,
            past_key_values=DynamicCache.from_legacy_cache(full_prefix_past),
            position_ids=next_position,
            cache_position=next_cache_position,
            use_cache=True,
            output_attentions=False,
        )
    stock_logits = stock_outputs.logits[:, -1, :]
    stock_token = int(torch.argmax(stock_logits, dim=-1).item())

    print("Patching layer", LAYER_TO_PATCH, "and running Triton streaming forward...")
    timings: list[float] = []
    layer_module = model.model.layers[LAYER_TO_PATCH].self_attn
    original_forward = layer_module.forward
    layer_module.forward = types.MethodType(
        build_triton_streaming_forward(store, LAYER_TO_PATCH, device, num_blocks, tail_past, timings), layer_module
    )

    # Warmup: first call pays Triton's JIT compile cost, which isn't representative of steady-state latency.
    with torch.inference_mode():
        model(
            input_ids=next_input_id,
            attention_mask=attention_mask,
            past_key_values=DynamicCache.from_legacy_cache(full_prefix_past),
            position_ids=next_position,
            cache_position=next_cache_position,
            use_cache=True,
            output_attentions=False,
        )
    timings.clear()

    try:
        with torch.inference_mode():
            streaming_outputs = model(
                input_ids=next_input_id,
                attention_mask=attention_mask,
                past_key_values=DynamicCache.from_legacy_cache(full_prefix_past),
                position_ids=next_position,
                cache_position=next_cache_position,
                use_cache=True,
                output_attentions=False,
            )
    finally:
        layer_module.forward = original_forward

    streaming_logits = streaming_outputs.logits[:, -1, :]
    streaming_token = int(torch.argmax(streaming_logits, dim=-1).item())

    max_logit_diff = (stock_logits.float() - streaming_logits.float()).abs().max().item()

    print("\nReal-layer Triton streaming attention comparison")
    print("--------------------------------------------------")
    print("stock_argmax_token:", stock_token, repr(tokenizer.decode([stock_token])))
    print("streaming_argmax_token:", streaming_token, repr(tokenizer.decode([streaming_token])))
    print("same_argmax_token:", stock_token == streaming_token)
    print("max_logit_diff:", f"{max_logit_diff:.6f}")
    print("triton_streaming_one_layer_step_ms (post-warmup):", f"{1000 * timings[0]:.4f}")

    assert stock_token == streaming_token, "Triton streaming layer diverged from stock at the argmax token"

    print("\nThroughput checkpoint against the existing (non-streaming) PagedModel...")
    paged_model = PagedModel(
        model,
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        recent_window=RECENT_WINDOW,
        policy="recent_only",
        tokens_per_block=STREAM_CHUNK_TOKENS,
    )
    warmup_token = paged_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1)
    extended_ids = torch.cat([input_ids, torch.tensor([warmup_token], device=device)], dim=1)
    extended_mask = torch.ones_like(extended_ids)

    torch.cuda.synchronize()
    started = time.perf_counter()
    paged_model.generate(
        input_ids=extended_ids, attention_mask=extended_mask, max_new_tokens=NEW_TOKENS_FOR_BASELINE_TIMING
    )
    torch.cuda.synchronize()
    existing_pagedmodel_step_ms = 1000 * (time.perf_counter() - started) / NEW_TOKENS_FOR_BASELINE_TIMING

    projected_all_layers_ms = timings[0] * 1000 * model.config.num_hidden_layers

    print("existing_pagedmodel_full_step_ms (all 24 layers):", f"{existing_pagedmodel_step_ms:.2f}")
    print(
        "projected_triton_attention_only_ms (this one layer's cost x num_layers, MLP/norms not even included):",
        f"{projected_all_layers_ms:.4f}",
    )
    print(f"projected_ratio_vs_existing_full_step: {projected_all_layers_ms / existing_pagedmodel_step_ms:.3f}x")

    print("\nOK: Triton streaming attention, hooked into one real layer, matches stock output exactly.")


if __name__ == "__main__":
    main()
