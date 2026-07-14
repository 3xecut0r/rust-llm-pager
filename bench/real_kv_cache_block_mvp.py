from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import pager
from torch_kv_block_store import KVBlockStore
from torch_kv_offload_mvp import bytes_to_mb, print_summary


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TOKENS_PER_BLOCK = 16
MAX_LENGTH = 192

VRAM_BUDGET = 128_000_000
RAM_BUDGET = 2_000_000_000

RECENT_WINDOW = 64
REBALANCE_INTERVAL = 4
PROMOTE_MARGIN = 0.05
RAM_PROMOTE_MARGIN = 0.20

POLICY = "sinks_heavy_hitter"


def format_block_list(block_ids: list[int], limit: int = 30) -> str:
    if len(block_ids) <= limit:
        return str(block_ids)

    shown = block_ids[:limit]
    remaining = len(block_ids) - limit

    return f"{shown} ... (+{remaining} more)"

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
    """
    Supports both legacy tuple past_key_values and newer cache objects.
    Returns list[(key, value)] where each tensor has shape:
        [batch, kv_heads, seq_len, head_dim]
    """
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
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Convert real model past_key_values into block tensors.

    Input per layer:
        key/value shape: [batch, kv_heads, seq_len, head_dim]

    Output per block:
        key/value shape: [layers, tokens_per_block, kv_heads, head_dim]

    Only full blocks are used in this MVP.
    """
    if not past_key_values:
        raise ValueError("past_key_values is empty.")

    first_key = past_key_values[0][0]
    if first_key.ndim != 4:
        raise ValueError(f"Expected KV tensor shape [batch, heads, seq, dim], got {first_key.shape}")

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
            # [batch, heads, seq, dim] -> [tokens, heads, dim]
            key_slice = layer_key[0, :, start:end, :].permute(1, 0, 2).contiguous()
            value_slice = layer_value[0, :, start:end, :].permute(1, 0, 2).contiguous()

            block_keys.append(key_slice)
            block_values.append(value_slice)

        # [layers, tokens, heads, dim]
        block_key = torch.stack(block_keys, dim=0).contiguous()
        block_value = torch.stack(block_values, dim=0).contiguous()

        blocks.append((block_key, block_value))

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
    """
    Build one block-level attention trace from real model attentions.

    We average:
    - across layers
    - across heads
    - for the last query token only
    """
    attentions = outputs.attentions

    if attentions is None:
        raise RuntimeError("Model did not return attentions.")

    traces = []

    for layer_attn in attentions:
        # shape: [batch, heads, query_len, key_len]
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
    return [x / total for x in out]


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

    past_key_values = get_legacy_past_key_values(outputs)

    kv_blocks = real_past_to_blocks(
        past_key_values,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    if not kv_blocks:
        raise RuntimeError("No full KV blocks were extracted. Increase prompt length or reduce TOKENS_PER_BLOCK.")

    store = KVBlockStore(tokens_per_block=TOKENS_PER_BLOCK)

    for block_id, (key, value) in enumerate(kv_blocks):
        store.put_gpu(block_id, key, value)

    real_block_size_mb = store.resident_gpu_bytes() / len(kv_blocks) / 1_000_000

    print("real_model_kv_blocks:", len(kv_blocks))
    print("real_model_kv_block_size_mb:", f"{real_block_size_mb:.3f}")

    print_summary("After real model KV extraction", store)

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
        num_blocks=len(kv_blocks),
    )

    query_block = len(kv_blocks) - 1

    # In this MVP, pager logical blocks are aligned with extracted KV block ids.
    p.on_step(query_block, 0, block_attention)

    tiers = p.tiers()
    movement = store.apply_tiers(tiers, device)

    print_summary("After Rust pager placement on real model KV", store)

    print("moved_to_gpu:", format_block_list(movement["to_gpu"]))
    print("moved_to_cpu:", format_block_list(movement["to_cpu"]))
    print("pager vram blocks:", p.vram_block_ids())
    print("store gpu blocks:", store.gpu_block_ids())
    real_attention_in_gpu = sum(
        block_attention[block_id]
        for block_id in store.gpu_block_ids()
        if block_id < len(block_attention)
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
