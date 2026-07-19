from __future__ import annotations

import torch
from torch_kv_block_store import KVBlockStore


def build_prompt() -> str:
    """Build the synthetic long-context prompt with an embedded needle fact."""
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

    question = "Question: What is the secret project codename mentioned at the beginning?\nAnswer:"

    return "\n\n".join([needle, filler, question])


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


def extract_single_block_from_past(
    past_key_values, *, block_id: int, tokens_per_block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract one block's (key, value) across all layers, batch size 1 only."""
    if not past_key_values:
        raise ValueError("past_key_values is empty.")

    first_key = past_key_values[0][0]

    if first_key.ndim != 4:
        raise ValueError(f"Expected KV tensor shape [batch, heads, seq, dim], got {first_key.shape}")

    batch, _, seq_len, _ = first_key.shape

    if batch != 1:
        raise ValueError("This MVP expects batch size 1.")

    start = block_id * tokens_per_block
    end = start + tokens_per_block

    if end > seq_len:
        raise ValueError(f"Block {block_id} is not fully available: end={end}, seq_len={seq_len}")

    block_keys = []
    block_values = []

    for layer_key, layer_value in past_key_values:
        block_keys.append(layer_key[0, :, start:end, :].permute(1, 0, 2).contiguous())
        block_values.append(layer_value[0, :, start:end, :].permute(1, 0, 2).contiguous())

    return torch.stack(block_keys, dim=0).contiguous(), torch.stack(block_values, dim=0).contiguous()


def real_past_to_blocks(
    past_key_values, *, tokens_per_block: int, verbose: bool = True
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split past_key_values into a list of full blocks, batch size 1 only."""
    if not past_key_values:
        raise ValueError("past_key_values is empty.")

    first_key = past_key_values[0][0]

    if first_key.ndim != 4:
        raise ValueError(f"Expected KV tensor shape [batch, heads, seq, dim], got {first_key.shape}")

    batch, kv_heads, seq_len, head_dim = first_key.shape

    if batch != 1:
        raise ValueError("This MVP expects batch size 1.")

    num_blocks = seq_len // tokens_per_block
    blocks = [
        extract_single_block_from_past(past_key_values, block_id=block_id, tokens_per_block=tokens_per_block)
        for block_id in range(num_blocks)
    ]

    if verbose:
        print("real_kv_num_layers:", len(past_key_values))
        print("real_kv_seq_len:", seq_len)
        print("real_kv_heads:", kv_heads)
        print("real_kv_head_dim:", head_dim)
        print("real_kv_full_blocks:", num_blocks)
        print("real_kv_ignored_tail_tokens:", seq_len - num_blocks * tokens_per_block)

    return blocks


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


def split_full_blocks_and_tail(past_key_values, *, tokens_per_block: int):
    """Split past_key_values into full tokens_per_block-sized blocks and a leftover tail."""
    full_tokens = (past_key_values[0][0].shape[2] // tokens_per_block) * tokens_per_block

    full_past = []
    tail_past = []

    for key, value in past_key_values:
        full_past.append((key[:, :, :full_tokens, :].contiguous(), value[:, :, :full_tokens, :].contiguous()))
        tail_past.append((key[:, :, full_tokens:, :].contiguous(), value[:, :, full_tokens:, :].contiguous()))

    return full_past, tail_past, full_tokens


def reconstruct_past_from_store(
    *, store: KVBlockStore, num_layers: int, num_blocks: int, tail_past=None
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Rebuild past_key_values from stored blocks, batch size 1 only, with an
    optional leftover tail (from split_full_blocks_and_tail) appended.

    Writes each block, and the tail, straight into one pre-allocated
    destination tensor sized for both up front, instead of building a
    blocks-only tensor and then torch.cat-ing the tail onto it: the old way
    needs the whole reconstructed cache resident twice at once (once as the
    blocks-only result, again as the cat output), this way needs it once.
    """
    sample_key, _ = store.get_gpu(0)
    tokens_per_block, kv_heads, head_dim = sample_key[0].shape
    tail_tokens = tail_past[0][0].shape[2] if tail_past else 0
    tail_start = num_blocks * tokens_per_block
    total_tokens = tail_start + tail_tokens

    dest = [
        (
            torch.empty(
                (1, kv_heads, total_tokens, head_dim),
                dtype=sample_key[layer_idx].dtype,
                device=sample_key[layer_idx].device,
            ),
            torch.empty(
                (1, kv_heads, total_tokens, head_dim),
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
            dest[layer_idx][0][:, :, start:end, :] = block_key[layer_idx].permute(1, 0, 2).unsqueeze(0)
            dest[layer_idx][1][:, :, start:end, :] = block_value[layer_idx].permute(1, 0, 2).unsqueeze(0)

    if tail_past:
        for layer_idx in range(num_layers):
            dest[layer_idx][0][:, :, tail_start:, :] = tail_past[layer_idx][0]
            dest[layer_idx][1][:, :, tail_start:, :] = tail_past[layer_idx][1]

    return dest


def build_store_from_blocks(
    kv_blocks: list[tuple[torch.Tensor, torch.Tensor]], *, tokens_per_block: int
) -> KVBlockStore:
    """Build a KVBlockStore from a list of (key, value) blocks, all placed on GPU."""
    store = KVBlockStore(tokens_per_block=tokens_per_block)

    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    return store


def format_block_list(block_ids: list[int], limit: int = 30) -> str:
    if len(block_ids) <= limit:
        return str(block_ids)

    return f"{block_ids[:limit]} ... (+{len(block_ids) - limit} more)"
