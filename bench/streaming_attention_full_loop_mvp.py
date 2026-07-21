from __future__ import annotations

import argparse
import types

import torch
from config import MODEL_NAME, TOKENS_PER_BLOCK
from real_kv_utils import get_legacy_past_key_values, real_past_to_blocks, split_full_blocks_and_tail
from real_kv_vram_savings_proof_mvp import build_long_prompt, chunked_prefill
from streaming_attention_triton_kernel_mvp import (
    streaming_attention_finalize,
    streaming_attention_state_init,
    streaming_attention_step,
)
from torch_kv_block_store import KVBlockStore
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

STREAM_CHUNK_TOKENS = TOKENS_PER_BLOCK
PREFILL_CHUNK = 512
GPU_RESIDENT_BLOCK_BUDGET = 2  # how many blocks the "pager" keeps warm on GPU; the rest live on CPU between steps
GROUP_SIZE_BLOCKS = 16  # how many blocks are gathered into GPU memory at once during attention -- bounds peak memory


class StreamingState:
    """Mutable state shared by every patched layer's forward closure across the whole decode loop."""

    def __init__(self, store: KVBlockStore, num_blocks: int, tail_past, num_layers: int):
        self.store = store
        self.num_blocks = num_blocks
        self.tail_past = tail_past  # list of (key, value) per layer, legacy [1, kv_heads, tail_len, head_dim]
        self.num_layers = num_layers


def gather_layer_kv_group(*, store: KVBlockStore, layer_idx: int, block_ids: list[int], device: torch.device):
    """
    Gather only the given block_ids into one small contiguous [kv_heads, group_tokens, head_dim]
    K/V tensor -- not the whole context. Called once per group, so peak GPU memory for this step
    is bounded by len(block_ids) blocks, not the total block count.
    """
    if not block_ids:
        return None, None

    store.ensure_gpu(block_ids[0], device)
    sample_key, _ = store.get_gpu(block_ids[0])
    block_len, kv_heads, head_dim = sample_key.shape[1], sample_key.shape[2], sample_key.shape[3]
    total_tokens = len(block_ids) * block_len

    dest_key = torch.empty((kv_heads, total_tokens, head_dim), dtype=sample_key.dtype, device=device)
    dest_value = torch.empty_like(dest_key)

    for i, block_id in enumerate(block_ids):
        store.ensure_gpu(block_id, device)
        block_key, block_value = store.get_gpu(block_id)
        start = i * block_len
        end = start + block_len
        dest_key[:, start:end, :] = block_key[layer_idx].permute(1, 0, 2)
        dest_value[:, start:end, :] = block_value[layer_idx].permute(1, 0, 2)

    return dest_key, dest_value


def build_layer_forward(state: StreamingState, layer_idx: int, device: torch.device):
    """One decode-time forward for one layer: gather store blocks + tail + new token, run the Triton kernel."""

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
        assert bsz == 1 and q_len == 1

        query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = (
            self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        # kv_seq_len sizes the rotary cos/sin table (it gets sliced to exactly this length), so it must reflect
        # the true absolute position being indexed by apply_rotary_pos_emb below -- NOT the small tail-only
        # cache's own length, which is deliberately much shorter than the real sequence position in this mode.
        kv_seq_len = int(position_ids.max().item()) + 1
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

        q = query_states[0, :, 0, :]
        m, l, acc = streaming_attention_state_init(q.shape[0], q.shape[1], device)

        # Process historical blocks GROUP_SIZE_BLOCKS at a time: only one group's worth of KV is
        # ever resident on GPU at once (freed before the next group is gathered), instead of the
        # whole context -- this is what actually bounds peak memory, not just the kernel speed.
        block_ids = list(range(state.num_blocks))
        for i in range(0, len(block_ids), GROUP_SIZE_BLOCKS):
            group_key, group_value = gather_layer_kv_group(
                store=state.store, layer_idx=layer_idx, block_ids=block_ids[i : i + GROUP_SIZE_BLOCKS], device=device
            )
            streaming_attention_step(q, group_key, group_value, m, l, acc)
            del group_key, group_value

        tail_key, tail_value = state.tail_past[layer_idx][0][0], state.tail_past[layer_idx][1][0]
        final_key = torch.cat([tail_key, key_states[0]], dim=1)
        final_value = torch.cat([tail_value, value_states[0]], dim=1)
        streaming_attention_step(q, final_key, final_value, m, l, acc)

        streaming_out = streaming_attention_finalize(acc, l, q.dtype)

        attn_output = streaming_out.unsqueeze(0).unsqueeze(2)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

    return forward


def patch_all_layers(model, state: StreamingState, device: torch.device):
    """Patch every decoder layer's self_attn.forward; return the originals so they can be restored."""
    originals = []
    for layer_idx, layer in enumerate(model.model.layers):
        originals.append(layer.self_attn.forward)
        layer.self_attn.forward = types.MethodType(build_layer_forward(state, layer_idx, device), layer.self_attn)
    return originals


def unpatch_all_layers(model, originals):
    for layer, original_forward in zip(model.model.layers, originals):
        layer.self_attn.forward = original_forward


def apply_tiers_keep_recent(store: KVBlockStore, num_blocks: int, device: torch.device) -> None:
    """Simplest possible placement: keep only the most recent GPU_RESIDENT_BLOCK_BUDGET blocks on GPU, rest on CPU."""
    warm_start = max(num_blocks - GPU_RESIDENT_BLOCK_BUDGET, 0)
    for block_id in range(num_blocks):
        if block_id < warm_start:
            store.ensure_cpu(block_id)
        else:
            store.ensure_gpu(block_id, device)


def streaming_generate(
    *, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, new_tokens: int, device, on_primed=None
):
    """
    Prime normally (stock, chunked), then decode new_tokens with every layer
    streaming from a KVBlockStore. on_primed(), if given, is called right
    after priming and before decoding starts -- priming always needs the
    full dense cache regardless of streaming mode, so peak-memory
    measurements must reset here to isolate the decode loop's own peak,
    not get dominated by the (identical either way) priming cost.
    """
    prefill_outputs = chunked_prefill(model, input_ids[:, :-1], attention_mask[:, :-1], PREFILL_CHUNK)
    full_prefix_past = get_legacy_past_key_values(prefill_outputs)

    full_past, tail_past, full_tokens = split_full_blocks_and_tail(
        full_prefix_past, tokens_per_block=STREAM_CHUNK_TOKENS
    )
    kv_blocks = real_past_to_blocks(full_past, tokens_per_block=STREAM_CHUNK_TOKENS, verbose=False)
    num_layers = len(full_prefix_past)

    store = KVBlockStore(tokens_per_block=STREAM_CHUNK_TOKENS)
    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    state = StreamingState(store=store, num_blocks=len(kv_blocks), tail_past=tail_past, num_layers=num_layers)
    apply_tiers_keep_recent(store, state.num_blocks, device)

    del prefill_outputs, full_prefix_past, full_past, kv_blocks
    torch.cuda.empty_cache()
    if on_primed is not None:
        on_primed()

    originals = patch_all_layers(model, state, device)

    next_input_id = input_ids[:, -1:]
    current_position = torch.tensor([[input_ids.shape[-1] - 1]], device=device)
    current_cache_position = torch.tensor([input_ids.shape[-1] - 1], device=device)
    current_mask = attention_mask
    generated: list[int] = []

    try:
        for _ in range(new_tokens):
            tail_cache = DynamicCache.from_legacy_cache(tuple(state.tail_past))

            with torch.inference_mode():
                outputs = model(
                    input_ids=next_input_id,
                    attention_mask=current_mask,
                    past_key_values=tail_cache,
                    position_ids=current_position,
                    cache_position=current_cache_position,
                    use_cache=True,
                    output_attentions=False,
                )

            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(next_token_id.item()))

            grown_tail = get_legacy_past_key_values(outputs)
            newly_full_past, remaining_tail, newly_full_tokens = split_full_blocks_and_tail(
                grown_tail, tokens_per_block=STREAM_CHUNK_TOKENS
            )
            if newly_full_tokens > 0:
                for block in real_past_to_blocks(newly_full_past, tokens_per_block=STREAM_CHUNK_TOKENS, verbose=False):
                    store.put_gpu(state.num_blocks, *block)
                    state.num_blocks += 1

            state.tail_past = remaining_tail
            apply_tiers_keep_recent(store, state.num_blocks, device)

            next_input_id = next_token_id
            current_position = current_position + 1
            current_cache_position = current_cache_position + 1
            current_mask = torch.cat(
                [current_mask, torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=device)], dim=1
            )
    finally:
        unpatch_all_layers(model, originals)

    return generated, state.num_blocks


def baseline_generate(*, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, new_tokens: int, on_primed=None):
    """Plain generation, full KV cache always resident on GPU, chunked prefill to avoid the logits OOM."""
    outputs = chunked_prefill(model, input_ids[:, :-1], attention_mask[:, :-1], PREFILL_CHUNK)

    if on_primed is not None:
        on_primed()

    next_input_id = input_ids[:, -1:]
    current_mask = attention_mask
    current_cache = outputs.past_key_values
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(new_tokens):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_mask,
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(next_token_id.item()))

            current_cache = outputs.past_key_values
            next_input_id = next_token_id
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=current_mask.device),
                ],
                dim=1,
            )

    return generated


def parse_args() -> argparse.Namespace:
    """Parse --context-tokens and --new-tokens."""
    parser = argparse.ArgumentParser(description="Stage D: full multi-step, all-layer Triton streaming generation.")
    parser.add_argument("--context-tokens", type=int, default=2000)
    parser.add_argument("--new-tokens", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")
    print("device:", device)
    print("model:", MODEL_NAME)
    print("context_tokens (target):", args.context_tokens)
    print("new_tokens:", args.new_tokens)
    print("gpu_resident_block_budget:", GPU_RESIDENT_BLOCK_BUDGET)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()

    prompt = build_long_prompt(tokenizer, args.context_tokens)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=args.context_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    print("actual_context_tokens:", input_ids.shape[-1])

    # Priming (chunked prefill) always needs the full dense cache, identically in both paths, so
    # its memory cost would dominate and hide any difference from the decode loop if peak were
    # measured across the whole call -- reset right after priming to isolate the decode-loop peak.
    print("\nRunning baseline (full KV resident on GPU)...")
    baseline_ids = baseline_generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        new_tokens=args.new_tokens,
        on_primed=lambda: torch.cuda.reset_peak_memory_stats(device),
    )
    baseline_decode_peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000
    torch.cuda.empty_cache()

    print("Running Triton streaming (all 24 layers, KVBlockStore-backed)...")
    streaming_ids, final_num_blocks = streaming_generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        new_tokens=args.new_tokens,
        device=device,
        on_primed=lambda: torch.cuda.reset_peak_memory_stats(device),
    )
    streaming_decode_peak_mb = torch.cuda.max_memory_allocated(device) / 1_000_000

    print("\nStage D summary")
    print("---------------")
    print("baseline_ids:", baseline_ids)
    print("streaming_ids:", streaming_ids)
    print("same_token_ids:", baseline_ids == streaming_ids)
    print("final_num_blocks:", final_num_blocks)
    print("baseline_decode_peak_gpu_mb (post-priming):", f"{baseline_decode_peak_mb:.2f}")
    print("streaming_decode_peak_gpu_mb (post-priming):", f"{streaming_decode_peak_mb:.2f}")
    print(f"decode_peak_reduction: {100.0 * (1.0 - streaming_decode_peak_mb / baseline_decode_peak_mb):.1f}%")

    assert baseline_ids == streaming_ids, "streaming generation diverged from baseline"

    print("\nOK: multi-step, all-layer Triton streaming generation matches baseline exactly.")


if __name__ == "__main__":
    main()
