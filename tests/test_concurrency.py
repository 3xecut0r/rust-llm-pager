from __future__ import annotations

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
    pm = make_paged_model(policy="heavy_hitter")  # needs_attention=True, so batch_size>1 is rejected

    with pytest.raises(NotImplementedError):
        pm.generate(
            input_ids=torch.zeros(2, 1, dtype=torch.long),
            attention_mask=torch.ones(2, 1, dtype=torch.long),
            max_new_tokens=1,
        )

    assert not pm._lock.locked()


def test_lock_is_released_after_reset():
    pm = make_paged_model()
    pm.reset()

    assert not pm._lock.locked()
