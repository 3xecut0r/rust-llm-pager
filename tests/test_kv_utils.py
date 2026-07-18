from __future__ import annotations

import torch

from pager_hf.kv_utils import (
    append_tail_to_reconstructed_past,
    extract_single_block_from_past,
    get_legacy_past_key_values,
    real_past_to_blocks,
    reconstruct_past_from_store,
    split_full_blocks_and_tail,
)


class FakeBlockStore:
    """
    Minimal stand-in for pager_hf.KVBlockStore that only implements what
    reconstruct_past_from_store needs. The real KVBlockStore requires CUDA
    tensors (it exists specifically to move blocks GPU <-> CPU), so these
    tests exercise the tensor-shape bookkeeping in kv_utils on CPU only.
    """

    def __init__(self):
        self._blocks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def put(self, block_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        self._blocks[block_id] = (key, value)

    def get_gpu(self, block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._blocks[block_id]


def make_fake_past(*, num_layers, batch, kv_heads, seq_len, head_dim, seed=0):
    generator = torch.Generator().manual_seed(seed)
    past = []
    for _ in range(num_layers):
        key = torch.randn(batch, kv_heads, seq_len, head_dim, generator=generator)
        value = torch.randn(batch, kv_heads, seq_len, head_dim, generator=generator)
        past.append((key, value))
    return past


def test_split_full_blocks_and_tail_splits_at_block_boundary():
    past = make_fake_past(num_layers=2, batch=1, kv_heads=2, seq_len=35, head_dim=4)

    full_past, tail_past, full_tokens = split_full_blocks_and_tail(
        past, tokens_per_block=16
    )

    assert full_tokens == 32
    assert full_past[0][0].shape[2] == 32
    assert tail_past[0][0].shape[2] == 3


def test_split_full_blocks_and_tail_handles_exact_multiple():
    past = make_fake_past(num_layers=1, batch=1, kv_heads=1, seq_len=32, head_dim=2)

    full_past, tail_past, full_tokens = split_full_blocks_and_tail(
        past, tokens_per_block=16
    )

    assert full_tokens == 32
    assert tail_past[0][0].shape[2] == 0


def _assert_extract_reconstruct_round_trip(batch: int):
    num_layers, kv_heads, head_dim, tokens_per_block = 3, 2, 4, 16
    seq_len = tokens_per_block * 4

    past = make_fake_past(
        num_layers=num_layers,
        batch=batch,
        kv_heads=kv_heads,
        seq_len=seq_len,
        head_dim=head_dim,
    )

    blocks = real_past_to_blocks(past, tokens_per_block=tokens_per_block)
    assert len(blocks) == 4

    for key, value in blocks:
        assert key.shape == (num_layers, batch, tokens_per_block, kv_heads, head_dim)
        assert value.shape == (num_layers, batch, tokens_per_block, kv_heads, head_dim)

    store = FakeBlockStore()
    for block_id, (key, value) in enumerate(blocks):
        store.put(block_id, key, value)

    reconstructed = reconstruct_past_from_store(
        store=store, num_layers=num_layers, num_blocks=4
    )

    assert len(reconstructed) == num_layers

    for layer_idx in range(num_layers):
        rec_key, rec_value = reconstructed[layer_idx]
        orig_key, orig_value = past[layer_idx]

        assert rec_key.shape == orig_key.shape == (batch, kv_heads, seq_len, head_dim)
        assert torch.equal(rec_key, orig_key)
        assert torch.equal(rec_value, orig_value)


def test_extract_and_reconstruct_round_trip_batch_1():
    _assert_extract_reconstruct_round_trip(batch=1)


def test_extract_and_reconstruct_round_trip_batch_3():
    _assert_extract_reconstruct_round_trip(batch=3)


def test_extract_single_block_from_past_selects_the_right_token_range():
    past = make_fake_past(num_layers=1, batch=1, kv_heads=1, seq_len=32, head_dim=2)

    key, value = extract_single_block_from_past(
        past, block_id=1, tokens_per_block=16
    )

    # block 1 covers tokens [16:32); after the permute+stack, shape is
    # [num_layers, batch, block_len, kv_heads, head_dim].
    expected_key = past[0][0][:, :, 16:32, :].permute(0, 2, 1, 3)
    assert torch.equal(key[0], expected_key)


def test_append_tail_to_reconstructed_past_concatenates_along_seq_dim():
    reconstructed = [(torch.zeros(1, 2, 32, 4), torch.zeros(1, 2, 32, 4))]
    tail = [(torch.ones(1, 2, 3, 4), torch.ones(1, 2, 3, 4))]

    out = append_tail_to_reconstructed_past(reconstructed, tail)

    key, value = out[0]
    assert key.shape == (1, 2, 35, 4)
    assert torch.equal(key[:, :, :32, :], torch.zeros(1, 2, 32, 4))
    assert torch.equal(key[:, :, 32:, :], torch.ones(1, 2, 3, 4))
    assert torch.equal(value[:, :, 32:, :], torch.ones(1, 2, 3, 4))


class _FakeOutputsTuple:
    def __init__(self, past):
        self.past_key_values = tuple(past)


class _FakeCache:
    def __init__(self, keys, values):
        self.key_cache = keys
        self.value_cache = values


class _FakeOutputsCacheObject:
    def __init__(self, past):
        self.past_key_values = _FakeCache(
            keys=[k for k, _ in past], values=[v for _, v in past]
        )


def test_get_legacy_past_key_values_accepts_tuple_form():
    past = make_fake_past(num_layers=2, batch=1, kv_heads=1, seq_len=4, head_dim=2)

    result = get_legacy_past_key_values(_FakeOutputsTuple(past))

    assert len(result) == 2
    assert torch.equal(result[0][0], past[0][0])
    assert torch.equal(result[1][1], past[1][1])


def test_get_legacy_past_key_values_accepts_cache_object_form():
    past = make_fake_past(num_layers=2, batch=1, kv_heads=1, seq_len=4, head_dim=2)

    result = get_legacy_past_key_values(_FakeOutputsCacheObject(past))

    assert len(result) == 2
    assert torch.equal(result[0][0], past[0][0])
    assert torch.equal(result[1][1], past[1][1])


def test_get_legacy_past_key_values_rejects_missing_cache():
    class _EmptyOutputs:
        past_key_values = None

    try:
        get_legacy_past_key_values(_EmptyOutputs())
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for missing past_key_values")
