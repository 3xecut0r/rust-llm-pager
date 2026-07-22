from __future__ import annotations

import pytest
import torch

from pager_hf.batched_decode import batched_decode_step
from pager_hf.paged_model import PagedModel

# All of batched_decode_step's validation runs before any real GPU
# computation, so these checks are testable on CPU with lightweight
# PagedModel instances whose session state is set directly instead of going
# through a real generate() call -- same "fake but sufficiently attributed"
# pattern already used for PagedModel's own locking tests (test_concurrency.py).


def make_fake_session(**overrides) -> PagedModel:
    pm = PagedModel(model=object(), vram_budget=1_000, ram_budget=1_000)
    pm._store = object()  # only needs to be non-None for these checks
    pm._num_layers = 2
    pm._num_blocks = 1
    pm._tail_past = [(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 4))]
    for key, value in overrides.items():
        setattr(pm, key, value)
    return pm


def test_empty_sessions_returns_empty_list():
    assert batched_decode_step([], [], torch.device("cpu")) == []


def test_rejects_sessions_with_different_underlying_models():
    a = make_fake_session()
    b = make_fake_session(model=object())
    with pytest.raises(ValueError, match="same underlying model"):
        batched_decode_step([a, b], [1, 2], torch.device("cpu"))


def test_rejects_non_streaming_sessions():
    a = make_fake_session()
    a.use_streaming_attention = False
    with pytest.raises(ValueError, match="use_streaming_attention=True"):
        batched_decode_step([a], [1], torch.device("cpu"))


def test_rejects_sessions_without_an_active_generate_session():
    a = make_fake_session()
    a._store = None
    with pytest.raises(ValueError, match="active generate\\(\\) session"):
        batched_decode_step([a], [1], torch.device("cpu"))


def test_rejects_mismatched_tokens_per_block():
    a = make_fake_session()
    b = make_fake_session(model=a.model, tokens_per_block=32)
    with pytest.raises(ValueError, match="tokens_per_block"):
        batched_decode_step([a, b], [1, 2], torch.device("cpu"))


def test_rejects_mismatched_streaming_group_size_blocks():
    a = make_fake_session()
    b = make_fake_session(model=a.model, streaming_group_size_blocks=999)
    with pytest.raises(ValueError, match="streaming_group_size_blocks"):
        batched_decode_step([a, b], [1, 2], torch.device("cpu"))


def test_rejects_batch_size_greater_than_one():
    a = make_fake_session()
    a._tail_past = [(torch.zeros(2, 2, 3, 4), torch.zeros(2, 2, 3, 4))]
    with pytest.raises(ValueError, match="single-row"):
        batched_decode_step([a], [1], torch.device("cpu"))


def test_rejects_mismatched_tail_length():
    a = make_fake_session()
    b = make_fake_session(model=a.model)
    b._tail_past = [(torch.zeros(1, 2, 7, 4), torch.zeros(1, 2, 7, 4))]
    with pytest.raises(ValueError, match="same tail length"):
        batched_decode_step([a, b], [1, 2], torch.device("cpu"))


def test_rejects_mixed_attention_scoring_policies():
    a = make_fake_session()
    a._needs_attention = True
    b = make_fake_session(model=a.model)
    b._needs_attention = False
    with pytest.raises(ValueError, match="same kind of policy"):
        batched_decode_step([a, b], [1, 2], torch.device("cpu"))
