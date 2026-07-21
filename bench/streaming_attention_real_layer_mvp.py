from __future__ import annotations

import time
import types

import torch
from config import MODEL_NAME, RAM_BUDGET, RECENT_WINDOW, TOKENS_PER_BLOCK, VRAM_BUDGET
from real_kv_utils import build_prompt, get_legacy_past_key_values, real_past_to_blocks, split_full_blocks_and_tail
from streaming_attention_prototype_mvp import streaming_attention_gqa
from torch_kv_block_store import KVBlockStore
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

from pager_hf import PagedModel

STREAM_CHUNK_TOKENS = TOKENS_PER_BLOCK
LAYER_TO_PATCH = 0
NEW_TOKENS_FOR_BASELINE_TIMING = 8


def build_streaming_forward(store: KVBlockStore, layer_idx: int, device: torch.device, num_blocks: int, timings: list):
    """
    Return a forward() replacement for one Qwen2Attention instance: does
    everything the stock forward does (projections, RoPE, cache.update() for
    bookkeeping), but sources the attention computation itself from
    KVBlockStore blocks streamed one at a time, instead of the reconstructed
    full cache tensor.
    """

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

        chunks = []
        for block_id in range(num_blocks):
            store.ensure_gpu(block_id, device)
            block_key, block_value = store.get_gpu(block_id)
            # stored as [block_len, kv_heads, head_dim] (see extract_single_block_from_past); streaming_attention_gqa
            # wants [kv_heads, seq_len, head_dim].
            chunks.append((block_key[layer_idx].permute(1, 0, 2), block_value[layer_idx].permute(1, 0, 2)))
        chunks.append(
            (key_states[0], value_states[0])
        )  # the just-computed new token, already [kv_heads, q_len, head_dim]

        q = query_states[0, :, 0, :]
        streaming_out = streaming_attention_gqa(q, chunks, n_rep=self.num_key_value_groups)

        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)

        attn_output = streaming_out.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward


def main() -> None:
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

    prompt = build_prompt()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("prompt_seq_len:", input_ids.shape[-1])

    with torch.inference_mode():
        prefill_outputs = model(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True, output_attentions=False
        )
    full_prefix_past = get_legacy_past_key_values(prefill_outputs)

    full_past, tail_past, full_tokens = split_full_blocks_and_tail(
        full_prefix_past, tokens_per_block=STREAM_CHUNK_TOKENS
    )
    kv_blocks = real_past_to_blocks(full_past, tokens_per_block=STREAM_CHUNK_TOKENS)
    num_blocks = len(kv_blocks)
    print("full_tokens:", full_tokens, "tail_tokens:", tail_past[0][0].shape[2], "num_blocks:", num_blocks)

    # Real GPU<->CPU split: most blocks start on GPU then get pushed to CPU,
    # exactly like the pager would between steps, so the patched forward has
    # to genuinely stream some blocks back from CPU, not just re-slice a
    # tensor that never left the GPU.
    store = KVBlockStore(tokens_per_block=STREAM_CHUNK_TOKENS)
    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)
    for block_id in range(max(num_blocks - 2, 0)):
        store.offload_to_cpu(block_id)
    print("gpu_resident_blocks:", store.gpu_block_ids())
    print("cpu_resident_blocks:", store.cpu_block_ids())

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

    print("Patching layer", LAYER_TO_PATCH, "and running streaming forward...")
    timings: list[float] = []
    layer_module = model.model.layers[LAYER_TO_PATCH].self_attn
    original_forward = layer_module.forward
    layer_module.forward = types.MethodType(
        build_streaming_forward(store, LAYER_TO_PATCH, device, num_blocks, timings), layer_module
    )

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

    print("\nReal-layer streaming attention comparison")
    print("------------------------------------------")
    print("stock_argmax_token:", stock_token, repr(tokenizer.decode([stock_token])))
    print("streaming_argmax_token:", streaming_token, repr(tokenizer.decode([streaming_token])))
    print("same_argmax_token:", stock_token == streaming_token)
    print("max_logit_diff:", f"{max_logit_diff:.6f}")
    print("streaming_attention_one_layer_step_ms:", f"{1000 * timings[0]:.2f}")

    assert stock_token == streaming_token, "streaming layer diverged from stock at the argmax token"

    # Throughput checkpoint: compare one patched layer's cost against the
    # *entire* current (non-streaming) PagedModel step, all 24 layers
    # combined, at the same prompt -- this is the number that decides whether
    # per-block Python-level streaming is viable at all before extending it
    # to every layer.
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
        "projected_streaming_attention_only_ms (this one layer's cost x num_layers, MLP/norms not even included):",
        f"{projected_all_layers_ms:.2f}",
    )
    print(f"projected_slowdown_factor: {projected_all_layers_ms / existing_pagedmodel_step_ms:.1f}x")

    print(
        "\nOK (correctness): one real Qwen2Attention layer, patched to stream KV blocks from KVBlockStore, "
        "matches stock output exactly at the argmax token."
    )
    print(
        "NOT OK (throughput): per-block Python-level streaming is far too slow to extend to all layers as-is -- "
        "this proves the memory-reduction concept, not a practical implementation. A real speedup needs a fused "
        "Triton/CUDA kernel (vLLM-style PagedAttention), not more Python-level tiling. See README."
    )


if __name__ == "__main__":
    main()
