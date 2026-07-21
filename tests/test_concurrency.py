from __future__ import annotations

import logging

import pytest
import torch

from pager_hf.paged_model import PagedModel


def make_paged_model(**overrides) -> PagedModel:
    """A PagedModel whose locking logic can be tested without a real model or CUDA."""
    kwargs = dict(model=object(), vram_budget=1_000, ram_budget=1_000)
    kwargs.update(overrides)
    return PagedModel(**kwargs)


def test_generate_raises_if_another_call_holds_the_lock():
    """Simulates a concurrent generate() call from another thread."""
    pm = make_paged_model()
    pm._lock.acquire()

    try:
        with pytest.raises(RuntimeError, match="already running"):
            pm.generate(
                input_ids=torch.zeros(1, 1, dtype=torch.long),
                attention_mask=torch.ones(1, 1, dtype=torch.long),
                max_new_tokens=1,
            )
    finally:
        pm._lock.release()


def test_lock_contention_is_logged(caplog):
    """Rejected concurrent calls should be visible in logs, not just as a raised exception."""
    pm = make_paged_model()
    pm._lock.acquire()

    try:
        with caplog.at_level(logging.WARNING, logger="pager_hf.paged_model"):
            with pytest.raises(RuntimeError, match="already running"):
                pm.generate(
                    input_ids=torch.zeros(1, 1, dtype=torch.long),
                    attention_mask=torch.ones(1, 1, dtype=torch.long),
                    max_new_tokens=1,
                )
    finally:
        pm._lock.release()

    assert any(r.levelno == logging.WARNING and "already in flight" in r.message for r in caplog.records)


def test_reset_raises_if_another_call_holds_the_lock():
    pm = make_paged_model()
    pm._lock.acquire()

    try:
        with pytest.raises(RuntimeError, match="already running"):
            pm.reset()
    finally:
        pm._lock.release()


def test_lock_is_released_after_generate_raises():
    """A validation error mid-call must not leave the instance permanently locked out."""
    pm = make_paged_model()

    with pytest.raises(NotImplementedError):
        pm.generate(
            input_ids=torch.zeros(1, 2, dtype=torch.long),
            attention_mask=torch.tensor([[1, 0]], dtype=torch.long),  # right-padded, rejected before touching the model
            max_new_tokens=1,
        )

    assert not pm._lock.locked()


def test_lock_is_released_after_reset():
    pm = make_paged_model()
    pm.reset()

    assert not pm._lock.locked()
