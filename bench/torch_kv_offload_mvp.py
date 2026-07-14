from __future__ import annotations

import torch

from torch_kv_block_store import KVBlockStore


TOKENS_PER_BLOCK = 16

NUM_BLOCKS = 64
NUM_LAYERS = 24
NUM_KV_HEADS = 8
HEAD_DIM = 64

DTYPE = torch.float16


def make_dummy_kv_block(
        *,
        device: torch.device,
        tokens_per_block: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (
        num_layers,
        tokens_per_block,
        num_kv_heads,
        head_dim,
    )

    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)

    return key, value


def bytes_to_mb(value: int) -> float:
    return value / 1_000_000


def print_summary(title: str, store: KVBlockStore) -> None:
    summary = store.summary()

    print(f"\n{title}")
    print("-" * len(title))
    print("gpu_blocks:", summary["gpu_blocks"])
    print("cpu_blocks:", summary["cpu_blocks"])
    print("resident_gpu_mb:", f"{bytes_to_mb(summary['resident_gpu_bytes']):.2f}")
    print("resident_cpu_mb:", f"{bytes_to_mb(summary['resident_cpu_bytes']):.2f}")
    print("gpu_to_cpu_mb:", f"{bytes_to_mb(summary['gpu_to_cpu_bytes']):.2f}")
    print("cpu_to_gpu_mb:", f"{bytes_to_mb(summary['cpu_to_gpu_bytes']):.2f}")
    print("gpu_to_cpu_copies:", summary["gpu_to_cpu_copies"])
    print("cpu_to_gpu_copies:", summary["cpu_to_gpu_copies"])
    print("gpu_to_cpu_sec:", f"{summary['gpu_to_cpu_sec']:.6f}")
    print("cpu_to_gpu_sec:", f"{summary['cpu_to_gpu_sec']:.6f}")

    print("gpu block ids first 20:", store.gpu_block_ids()[:20])
    print("cpu block ids first 20:", store.cpu_block_ids()[:20])


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("tokens_per_block:", TOKENS_PER_BLOCK)
    print("num_blocks:", NUM_BLOCKS)
    print("num_layers:", NUM_LAYERS)
    print("num_kv_heads:", NUM_KV_HEADS)
    print("head_dim:", HEAD_DIM)
    print("dtype:", DTYPE)

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

    print_summary("After initial GPU allocation", store)

    # Simulate eviction: keep only first 16 blocks on GPU.
    for block_id in range(16, NUM_BLOCKS):
        store.offload_to_cpu(block_id)

    print_summary("After GPU -> CPU offload", store)

    # Simulate prefetch: bring several blocks back to GPU.
    for block_id in [20, 21, 22, 40, 41, 42, 60, 61]:
        store.load_to_gpu(block_id, device)

    print_summary("After CPU -> GPU reload", store)

    expected_gpu_blocks = 16 + 8
    expected_cpu_blocks = NUM_BLOCKS - expected_gpu_blocks

    assert len(store.gpu_blocks) == expected_gpu_blocks
    assert len(store.cpu_blocks) == expected_cpu_blocks

    print("\nOK: real KV-like tensors moved GPU <-> CPU.")


if __name__ == "__main__":
    main()
    