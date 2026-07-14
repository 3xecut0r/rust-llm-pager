from __future__ import annotations

import random

import torch

import pager
from torch_kv_block_store import KVBlockStore
from torch_kv_offload_mvp import (
    DTYPE,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_LAYERS,
    TOKENS_PER_BLOCK,
    bytes_to_mb,
    make_dummy_kv_block,
    print_summary,
)


NUM_BLOCKS = 160

VRAM_BUDGET = 256_000_000
RAM_BUDGET = 2_000_000_000

RECENT_WINDOW = 64
REBALANCE_INTERVAL = 8
PROMOTE_MARGIN = 0.05
RAM_PROMOTE_MARGIN = 0.20

POLICY = "sinks_heavy_hitter"


def format_block_list(block_ids: list[int], limit: int = 30) -> str:
    if len(block_ids) <= limit:
        return str(block_ids)

    shown = block_ids[:limit]
    remaining = len(block_ids) - limit

    return f"{shown} ... (+{remaining} more)"

def make_attention_trace(
        *,
        query_block: int,
        num_blocks: int,
) -> list[float]:
    """
    Synthetic attention trace with three patterns:
    - sinks near the beginning
    - recent blocks near query position
    - a few stable heavy-hitter blocks in the middle
    """
    attn = [0.001 for _ in range(num_blocks)]

    # Sink attention.
    for block_id in range(min(4, num_blocks)):
        attn[block_id] += 0.20

    # Recent attention.
    recent_start = max(0, query_block - 8)
    for block_id in range(recent_start, min(num_blocks, query_block + 1)):
        attn[block_id] += 0.08

    # Stable heavy hitters.
    for block_id in [20, 21, 55, 56, 90, 91]:
        if block_id < num_blocks:
            attn[block_id] += 0.12

    # Small noise.
    for block_id in range(num_blocks):
        attn[block_id] += random.random() * 0.002

    total = sum(attn)
    return [x / total for x in attn]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    random.seed(42)

    device = torch.device("cuda")

    print("device:", device)
    print("policy:", POLICY)
    print("num_blocks:", NUM_BLOCKS)
    print("tokens_per_block:", TOKENS_PER_BLOCK)
    print("pager_vram_budget_mb:", bytes_to_mb(VRAM_BUDGET))
    print("pager_ram_budget_mb:", bytes_to_mb(RAM_BUDGET))
    print("pager_logical_block_size_mb:", 16.0)
    print("note: pager uses simulated 16 MB logical blocks")

    store = KVBlockStore(tokens_per_block=TOKENS_PER_BLOCK)

    for block_id in range(NUM_BLOCKS):
        key, value = make_dummy_kv_block(
            device=device,
            tokens_per_block=TOKENS_PER_BLOCK,
            num_layers=NUM_LAYERS,
            num_kv_heads=NUM_KV_HEADS,
            head_dim=HEAD_DIM,
            dtype=DTYPE,
        )
        store.put_gpu(block_id, key, value)

    real_block_size_mb = store.resident_gpu_bytes() / NUM_BLOCKS / 1_000_000
    print("real_dummy_kv_block_size_mb:", f"{real_block_size_mb:.3f}")
    print("note: pager budget controls logical placement; store reports real tensor bytes")

    p = pager.PyPager(
        VRAM_BUDGET,
        RAM_BUDGET,
        RECENT_WINDOW,
        REBALANCE_INTERVAL,
        PROMOTE_MARGIN,
        RAM_PROMOTE_MARGIN,
        POLICY,
    )

    print_summary("Initial all-GPU KV blocks", store)

    pending_to_gpu: list[int] = []
    pending_to_cpu: list[int] = []

    # Simulate decode steps.
    for step, query_block in enumerate(range(32, NUM_BLOCKS, 8), start=1):
        attn = make_attention_trace(
            query_block=query_block,
            num_blocks=query_block + 1,
        )

        # In this MVP, pager logical blocks are aligned with store block ids.
        token_idx = query_block

        p.on_step(token_idx, 0, attn)

        tiers = p.tiers()
        movement = store.apply_tiers(tiers, device)
        pending_to_gpu.extend(movement["to_gpu"])
        pending_to_cpu.extend(movement["to_cpu"])

        if step % 4 == 0:
            print_summary(f"After pager step {step}, query_block={query_block}", store)
            print(
                "moved_to_gpu_since_last_report:",
                format_block_list(pending_to_gpu),
            )
            print(
                "moved_to_cpu_since_last_report:",
                format_block_list(pending_to_cpu),
            )

            pending_to_gpu.clear()
            pending_to_cpu.clear()
            print("pager vram blocks first 30:", p.vram_block_ids()[:30])
            print("store gpu blocks first 30:", store.gpu_block_ids()[:30])

    print_summary("Final pager-controlled placement", store)

    metrics = p.metrics()

    print("\nPager metrics")
    print("-------------")
    print("tokens:", metrics.tokens)
    print("vram_peak_mb:", bytes_to_mb(metrics.vram_peak))
    print("ram_peak_mb:", bytes_to_mb(metrics.ram_peak))
    print("swap_vram_ram_mb:", bytes_to_mb(metrics.swap_vram_ram))
    print("swap_ram_ssd_mb:", bytes_to_mb(metrics.swap_ram_ssd))
    print("attention_mass_total:", f"{metrics.attention_mass_total:.4f}")
    print("attention_mass_vram:", f"{metrics.attention_mass_vram:.4f}")
    print(
        "vram_attention_ratio:",
        f"{metrics.attention_mass_vram / metrics.attention_mass_total:.4f}",
    )

    print("\nOK: Rust pager controlled real GPU <-> CPU KV-like tensor placement.")


if __name__ == "__main__":
    main()
