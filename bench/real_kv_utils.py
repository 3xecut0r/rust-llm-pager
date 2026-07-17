from __future__ import annotations

import torch

from torch_kv_block_store import KVBlockStore


def build_prompt() -> str:
    needle = (
        "IMPORTANT FACT: The secret project codename is BLUE ORCHID. "
        "Remember this codename because it will be asked later."
    )

    filler = """
This paragraph is unrelated filler text about software engineering,
memory management, operating systems, compilers, databases, networking,
and performance optimization. It mentions Rust, Python, Linux, GPUs,
caches, filesystems, and distributed systems, but it does not contain
the secret project codename.

Another unrelated paragraph describes how developers build services,
debug production issues, write benchmarks, profile latency, optimize
memory usage, and reason about trade-offs between throughput and quality.
This paragraph is intentionally noisy and should distract attention from
the important fact at the beginning.
""".strip()

    question = (
        "Question: What is the secret project codename mentioned at the beginning?\n"
        "Answer:"
    )

    return "\n\n".join([needle, filler, question])


def get_legacy_past_key_values(outputs):
    past = outputs.past_key_values

    if past is None:
        raise RuntimeError("Model did not return past_key_values.")

    if isinstance(past, (tuple, list)):
        return list(past)

    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return list(zip(past.key_cache, past.value_cache))

    raise TypeError(f"Unsupported past_key_values type: {type(past)!r}")


def real_past_to_blocks(
        past_key_values,
        *,
        tokens_per_block: int,
        verbose: bool = True,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if not past_key_values:
        raise ValueError("past_key_values is empty.")

    first_key = past_key_values[0][0]

    if first_key.ndim != 4:
        raise ValueError(
            f"Expected KV tensor shape [batch, heads, seq, dim], got {first_key.shape}"
        )

    batch, kv_heads, seq_len, head_dim = first_key.shape

    if batch != 1:
        raise ValueError("This MVP expects batch size 1.")

    num_layers = len(past_key_values)
    num_blocks = seq_len // tokens_per_block

    blocks = []

    for block_id in range(num_blocks):
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

        blocks.append((block_key, block_value))

    if verbose:
        print("real_kv_num_layers:", num_layers)
        print("real_kv_seq_len:", seq_len)
        print("real_kv_heads:", kv_heads)
        print("real_kv_head_dim:", head_dim)
        print("real_kv_full_blocks:", num_blocks)
        print("real_kv_ignored_tail_tokens:", seq_len - num_blocks * tokens_per_block)

    return blocks


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


def build_store_from_blocks(
        kv_blocks: list[tuple[torch.Tensor, torch.Tensor]],
        *,
        tokens_per_block: int,
) -> KVBlockStore:
    store = KVBlockStore(tokens_per_block=tokens_per_block)

    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    return store


def format_block_list(block_ids: list[int], limit: int = 30) -> str:
    if len(block_ids) <= limit:
        return str(block_ids)

    shown = block_ids[:limit]
    remaining = len(block_ids) - limit

    return f"{shown} ... (+{remaining} more)"

def extract_single_block_from_past(
        past_key_values,
        *,
        block_id: int,
        tokens_per_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not past_key_values:
        raise ValueError("past_key_values is empty.")

    first_key = past_key_values[0][0]

    if first_key.ndim != 4:
        raise ValueError(
            f"Expected KV tensor shape [batch, heads, seq, dim], got {first_key.shape}"
        )

    batch, _, seq_len, _ = first_key.shape

    if batch != 1:
        raise ValueError("This MVP expects batch size 1.")

    start = block_id * tokens_per_block
    end = start + tokens_per_block

    if end > seq_len:
        raise ValueError(
            f"Block {block_id} is not fully available: end={end}, seq_len={seq_len}"
        )

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
