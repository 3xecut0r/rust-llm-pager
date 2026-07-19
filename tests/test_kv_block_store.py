from __future__ import annotations

import pytest
import torch

from pager_hf.kv_block_store import KVBlockStore, kv_nbytes

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="KVBlockStore requires CUDA tensors by design.")


def make_block(*, num_layers=2, batch=1, tokens_per_block=16, kv_heads=2, head_dim=64):
    shape = (num_layers, batch, tokens_per_block, kv_heads, head_dim)
    device = torch.device("cuda")
    return torch.zeros(shape, dtype=torch.float16, device=device), torch.zeros(
        shape, dtype=torch.float16, device=device
    )


def test_offload_to_cpu_raises_when_ram_budget_would_be_exceeded():
    """A tiny ram_budget must reject the offload instead of growing CPU memory unbounded."""
    store = KVBlockStore(tokens_per_block=16, ram_budget_bytes=1000)
    key, value = make_block()
    store.put_gpu(0, key, value)

    with pytest.raises(RuntimeError, match="ram_budget exceeded"):
        store.ensure_cpu(0)

    assert store.has_gpu(0)
    assert not store.has_cpu(0)
    assert store.resident_cpu_bytes() == 0


def test_offload_to_cpu_succeeds_within_ram_budget():
    """A generous ram_budget should not block ordinary offloading."""
    store = KVBlockStore(tokens_per_block=16, ram_budget_bytes=1_000_000_000)
    key, value = make_block()
    store.put_gpu(0, key, value)

    store.ensure_cpu(0)

    assert store.has_cpu(0)
    assert not store.has_gpu(0)


def test_offload_to_cpu_ignores_ram_budget_when_unset():
    """ram_budget_bytes=None (the default) means no cap, matching the old unconditional behavior."""
    store = KVBlockStore(tokens_per_block=16)
    key, value = make_block()
    store.put_gpu(0, key, value)

    store.ensure_cpu(0)

    assert store.has_cpu(0)


def brute_force_resident_bytes(store: KVBlockStore) -> tuple[int, int]:
    """Recompute resident bytes from scratch, to check the incrementally tracked totals don't drift."""
    gpu_bytes = sum(kv_nbytes(key, value) for key, value in store.gpu_blocks.values())
    cpu_bytes = sum(kv_nbytes(key, value) for key, value in store.cpu_blocks.values())
    return gpu_bytes, cpu_bytes


def test_resident_bytes_stay_correct_across_many_moves():
    """The incrementally tracked totals (put_gpu/offload_to_cpu/load_to_gpu) must match a from-scratch sum."""
    store = KVBlockStore(tokens_per_block=16)
    device = torch.device("cuda")

    for block_id in range(10):
        key, value = make_block(tokens_per_block=8 + block_id)  # varying sizes, not all identical
        store.put_gpu(block_id, key, value)

    for block_id in range(0, 10, 2):
        store.offload_to_cpu(block_id)

    for block_id in range(0, 10, 4):
        store.load_to_gpu(block_id, device)

    expected_gpu_bytes, expected_cpu_bytes = brute_force_resident_bytes(store)
    assert store.resident_gpu_bytes() == expected_gpu_bytes
    assert store.resident_cpu_bytes() == expected_cpu_bytes
