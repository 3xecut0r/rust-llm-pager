from __future__ import annotations

import logging

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


def test_offload_to_cpu_logs_error_when_ram_budget_exceeded(caplog):
    """A rejected offload should also be visible in logs, not just as a raised exception."""
    store = KVBlockStore(tokens_per_block=16, ram_budget_bytes=1000)
    key, value = make_block()
    store.put_gpu(0, key, value)

    with caplog.at_level(logging.ERROR, logger="pager_hf.kv_block_store"):
        with pytest.raises(RuntimeError, match="ram_budget exceeded"):
            store.ensure_cpu(0)

    assert any(r.levelno == logging.ERROR and "ram_budget exceeded" in r.message for r in caplog.records)


def test_offload_to_cpu_warns_when_approaching_ram_budget(caplog):
    """Crossing the warning threshold without exceeding the budget should log, not raise."""
    key, value = make_block()
    block_bytes = kv_nbytes(key, value)
    store = KVBlockStore(tokens_per_block=16, ram_budget_bytes=int(block_bytes / 0.82))  # ~82% usage after one block
    store.put_gpu(0, key, value)

    with caplog.at_level(logging.WARNING, logger="pager_hf.kv_block_store"):
        store.ensure_cpu(0)

    assert store.has_cpu(0)
    assert any(r.levelno == logging.WARNING and "ram_budget usage" in r.message for r in caplog.records)


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


def make_row_tagged_block(*, num_layers=2, batch=4, tokens_per_block=16, kv_heads=2, head_dim=8):
    """A block whose every row is filled with its own row index (key) / row index + 1000 (value),
    so reordering the batch dimension is verifiable by content, not just shape."""
    device = torch.device("cuda")
    shape = (num_layers, batch, tokens_per_block, kv_heads, head_dim)
    key = torch.zeros(shape, dtype=torch.float16, device=device)
    value = torch.zeros(shape, dtype=torch.float16, device=device)
    for row in range(batch):
        key[:, row] = row
        value[:, row] = row + 1000
    return key, value


def test_reorder_batch_rows_reorders_duplicates_and_drops_on_both_tiers():
    """Beam search needs all three at once: a surviving beam's row can move to a new
    position (reorder), spawn more than one child (duplicate), or a losing beam's row can
    simply never appear in the new indices (drop) -- and this must work whichever tier a
    block currently lives on, without changing that tier."""
    store = KVBlockStore(tokens_per_block=16)

    gpu_key, gpu_value = make_row_tagged_block()
    cpu_key, cpu_value = make_row_tagged_block()
    store.put_gpu(0, gpu_key, gpu_value)
    store.put_gpu(1, cpu_key, cpu_value)
    store.offload_to_cpu(1)

    # new row 0 <- old row 2, new row 1 <- old row 2 (duplicate), new row 2 <- old row 1,
    # new row 3 <- old row 3. Old row 0 never appears (dropped).
    new_row_indices = [2, 2, 1, 3]
    store.reorder_batch_rows(new_row_indices)

    reordered_gpu_key, reordered_gpu_value = store.get_gpu(0)
    assert store.has_gpu(0)  # tier unchanged
    for new_row, old_row in enumerate(new_row_indices):
        assert torch.all(reordered_gpu_key[:, new_row] == old_row)
        assert torch.all(reordered_gpu_value[:, new_row] == old_row + 1000)

    assert store.has_cpu(1)  # tier unchanged
    reordered_cpu_key, reordered_cpu_value = store.cpu_blocks[1]
    for new_row, old_row in enumerate(new_row_indices):
        assert torch.all(reordered_cpu_key[:, new_row] == old_row)
        assert torch.all(reordered_cpu_value[:, new_row] == old_row + 1000)
