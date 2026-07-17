from __future__ import annotations

import torch

from .kv_block_store import KVBlockStore


def get_legacy_past_key_values(outputs):
    past = outputs.past_key_values

    if past is None:
        raise RuntimeError("Model did not return past_key_values.")

    if isinstance(past, (tuple, list)):
        return list(past)

    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return list(zip(past.key_cache, past.value_cache))

    raise TypeError(f"Unsupported past_key_values type: {type(past)!r}")


def split_full_blocks_and_tail(
        past_key_values,
        *,
        tokens_per_block: int,
):
    seq_len = past_key_values[0][0].shape[2]
    full_tokens = (seq_len // tokens_per_block) * tokens_per_block

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


def extract_single_block_from_past(
        past_key_values,
        *,
        block_id: int,
        tokens_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    start = block_id * tokens_per_block
    end = start + tokens_per_block

    block_keys = []
    block_values = []

    for layer_key, layer_value in past_key_values:
        key_slice = layer_key[0, :, start:end, :].permute(1, 0, 2).contiguous()
        value_slice = layer_value[0, :, start:end, :].permute(1, 0, 2).contiguous()

        block_keys.append(key_slice)
        block_values.append(value_slice)

    block_key = torch.stack(block_keys, dim=0).contiguous()
    block_value = torch.stack(block_values, dim=0).contiguous()

    return block_key, block_value


def real_past_to_blocks(
        past_key_values,
        *,
        tokens_per_block: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    seq_len = past_key_values[0][0].shape[2]
    num_blocks = seq_len // tokens_per_block

    return [
        extract_single_block_from_past(
            past_key_values,
            block_id=block_id,
            tokens_per_block=tokens_per_block,
        )
        for block_id in range(num_blocks)
    ]


def token_attention_to_blocks(
        attn_to_keys: torch.Tensor,
        *,
        tokens_per_block: int,
        num_blocks: int,
) -> list[float]:
    out = [0.0 for _ in range(num_blocks)]

    max_tokens = num_blocks * tokens_per_block

    for token_idx in range(max_tokens):
        block_idx = token_idx // tokens_per_block
        out[block_idx] += float(attn_to_keys[token_idx].item())

    total = sum(out)

    if total <= 0:
        return [1.0 / num_blocks for _ in range(num_blocks)]

    return [x / total for x in out]


def extract_last_query_block_attention(
        outputs,
        *,
        tokens_per_block: int,
        num_blocks: int,
) -> list[float]:
    attentions = outputs.attentions

    if attentions is None:
        raise RuntimeError("Model did not return attentions.")

    traces = []

    for layer_attn in attentions:
        layer_attn = layer_attn.detach().float().cpu()

        last_query = layer_attn[0, :, -1, :]
        attn_to_keys = last_query.mean(dim=0)

        block_attn = token_attention_to_blocks(
            attn_to_keys,
            tokens_per_block=tokens_per_block,
            num_blocks=num_blocks,
        )

        traces.append(block_attn)

    out = [0.0 for _ in range(num_blocks)]

    for trace in traces:
        for idx, value in enumerate(trace):
            out[idx] += value

    out = [x / len(traces) for x in out]

    total = sum(out)

    if total <= 0:
        return [1.0 / num_blocks for _ in range(num_blocks)]

    return [x / total for x in out]


def reconstruct_past_from_store(
        *,
        store: KVBlockStore,
        num_layers: int,
        num_blocks: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    layer_keys: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]
    layer_values: list[list[torch.Tensor]] = [[] for _ in range(num_layers)]

    for block_id in range(num_blocks):
        block_key, block_value = store.get_gpu(block_id)

        for layer_idx in range(num_layers):
            key_slice = block_key[layer_idx]
            value_slice = block_value[layer_idx]

            layer_keys[layer_idx].append(
                key_slice.permute(1, 0, 2).contiguous()
            )
            layer_values[layer_idx].append(
                value_slice.permute(1, 0, 2).contiguous()
            )

    reconstructed = []

    for layer_idx in range(num_layers):
        key = torch.cat(layer_keys[layer_idx], dim=1)
        value = torch.cat(layer_values[layer_idx], dim=1)

        key = key.unsqueeze(0).contiguous()
        value = value.unsqueeze(0).contiguous()

        reconstructed.append((key, value))

    return reconstructed


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
