from __future__ import annotations

import torch

from .kv_block_store import KVBlockStore


def get_legacy_past_key_values(outputs):
    """Normalize past_key_values into a plain list of (key, value) tuples."""
    past = outputs.past_key_values

    if past is None:
        raise RuntimeError("Model did not return past_key_values.")

    if isinstance(past, (tuple, list)):
        return list(past)

    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return list(zip(past.key_cache, past.value_cache))

    raise TypeError(f"Unsupported past_key_values type: {type(past)!r}")


def split_full_blocks_and_tail(past_key_values, *, tokens_per_block: int):
    """Split past_key_values into full tokens_per_block-sized blocks and a leftover tail."""
    full_tokens = (past_key_values[0][0].shape[2] // tokens_per_block) * tokens_per_block

    full_past = []
    tail_past = []

    for key, value in past_key_values:
        full_past.append((key[:, :, :full_tokens, :].contiguous(), value[:, :, :full_tokens, :].contiguous()))
        tail_past.append((key[:, :, full_tokens:, :].contiguous(), value[:, :, full_tokens:, :].contiguous()))

    return full_past, tail_past, full_tokens


def extract_single_block_from_past(
    past_key_values, *, block_id: int, tokens_per_block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one block's (key, value) across all layers, batch dimension included."""
    start = block_id * tokens_per_block
    end = start + tokens_per_block

    block_keys = []
    block_values = []

    for layer_key, layer_value in past_key_values:
        # [batch, kv_heads, block_len, head_dim] -> [batch, block_len, kv_heads, head_dim]
        block_keys.append(layer_key[:, :, start:end, :].permute(0, 2, 1, 3).contiguous())
        block_values.append(layer_value[:, :, start:end, :].permute(0, 2, 1, 3).contiguous())

    return torch.stack(block_keys, dim=0).contiguous(), torch.stack(block_values, dim=0).contiguous()


def real_past_to_blocks(past_key_values, *, tokens_per_block: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split past_key_values into a list of full blocks."""
    return [
        extract_single_block_from_past(past_key_values, block_id=block_id, tokens_per_block=tokens_per_block)
        for block_id in range(past_key_values[0][0].shape[2] // tokens_per_block)
    ]


def token_attention_to_blocks(attn_to_keys: torch.Tensor, *, tokens_per_block: int, num_blocks: int) -> list[float]:
    """Sum per-token attention into per-block attention, normalized to sum to 1."""
    out = [0.0] * num_blocks

    for token_idx in range(num_blocks * tokens_per_block):
        out[token_idx // tokens_per_block] += float(attn_to_keys[token_idx].item())

    total = sum(out)

    if total <= 0:
        return [1.0 / num_blocks] * num_blocks

    return [x / total for x in out]


def extract_last_query_block_attention(outputs, *, tokens_per_block: int, num_blocks: int) -> list[float]:
    """Average the last query token's attention to each block, across layers and heads."""
    attentions = outputs.attentions

    if attentions is None:
        raise RuntimeError("Model did not return attentions.")

    traces = [
        token_attention_to_blocks(
            layer_attn.detach().float().cpu()[0, :, -1, :].mean(dim=0),
            tokens_per_block=tokens_per_block,
            num_blocks=num_blocks,
        )
        for layer_attn in attentions
    ]

    out = [sum(values) / len(traces) for values in zip(*traces)]
    total = sum(out)

    if total <= 0:
        return [1.0 / num_blocks] * num_blocks

    return [x / total for x in out]


def reconstruct_past_from_store(
    *, store: KVBlockStore, num_layers: int, num_blocks: int, tail_past=None
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Rebuild past_key_values from stored blocks, inverse of extract_single_block_from_past,
    with an optional leftover tail (from split_full_blocks_and_tail) appended.

    Writes each block, and the tail, straight into one pre-allocated
    destination tensor sized for both up front, instead of building a
    blocks-only tensor and then torch.cat-ing the tail onto it: the old way
    needs the whole reconstructed cache resident twice at once (once as the
    blocks-only result, again as the cat output), this way needs it once.
    """
    sample_key, _ = store.get_gpu(0)
    batch_size, tokens_per_block, kv_heads, head_dim = sample_key[0].shape
    tail_tokens = tail_past[0][0].shape[2] if tail_past else 0
    tail_start = num_blocks * tokens_per_block
    total_tokens = tail_start + tail_tokens

    dest = [
        (
            torch.empty(
                (batch_size, kv_heads, total_tokens, head_dim),
                dtype=sample_key[layer_idx].dtype,
                device=sample_key[layer_idx].device,
            ),
            torch.empty(
                (batch_size, kv_heads, total_tokens, head_dim),
                dtype=sample_key[layer_idx].dtype,
                device=sample_key[layer_idx].device,
            ),
        )
        for layer_idx in range(num_layers)
    ]

    for block_id in range(num_blocks):
        block_key, block_value = store.get_gpu(block_id)
        start = block_id * tokens_per_block
        end = start + tokens_per_block

        for layer_idx in range(num_layers):
            # [batch, block_len, kv_heads, head_dim] -> [batch, kv_heads, block_len, head_dim]
            dest[layer_idx][0][:, :, start:end, :] = block_key[layer_idx].permute(0, 2, 1, 3)
            dest[layer_idx][1][:, :, start:end, :] = block_value[layer_idx].permute(0, 2, 1, 3)

    if tail_past:
        for layer_idx in range(num_layers):
            dest[layer_idx][0][:, :, tail_start:, :] = tail_past[layer_idx][0]
            dest[layer_idx][1][:, :, tail_start:, :] = tail_past[layer_idx][1]

    return dest
