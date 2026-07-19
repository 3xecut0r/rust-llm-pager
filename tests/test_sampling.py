from __future__ import annotations

import torch

from pager_hf.paged_model import PagedModel


def test_top_k_1_always_matches_argmax():
    """top_k=1 has only one token left to sample, so it must degenerate to greedy."""
    logits = torch.randn(5, 100)
    generator = torch.Generator().manual_seed(0)

    for row in range(5):
        sampled = PagedModel._sample_next_token(
            logits[row : row + 1], temperature=1.0, top_k=1, top_p=None, generator=generator
        )
        assert sampled.item() == torch.argmax(logits[row]).item()


def test_sampling_is_reproducible_with_a_seeded_generator():
    logits = torch.tensor([[5.0, 1.0, 0.5, 0.1, -2.0]])

    first = PagedModel._sample_next_token(
        logits, temperature=1.0, top_k=None, top_p=None, generator=torch.Generator().manual_seed(42)
    )
    second = PagedModel._sample_next_token(
        logits, temperature=1.0, top_k=None, top_p=None, generator=torch.Generator().manual_seed(42)
    )

    assert torch.equal(first, second)


def test_top_p_excludes_low_probability_tail():
    """A very small top_p should only ever leave the single dominant token in the nucleus."""
    logits = torch.tensor([[10.0, -10.0, -10.0, -10.0, -10.0]])
    generator = torch.Generator().manual_seed(0)

    for _ in range(10):
        sampled = PagedModel._sample_next_token(logits, temperature=1.0, top_k=None, top_p=0.01, generator=generator)
        assert sampled.item() == 0


def test_top_k_restricts_sampling_to_the_top_tokens():
    logits = torch.tensor([[5.0, 4.0, 3.0, -10.0, -10.0]])
    generator = torch.Generator().manual_seed(0)

    for _ in range(20):
        sampled = PagedModel._sample_next_token(logits, temperature=1.0, top_k=2, top_p=None, generator=generator)
        assert sampled.item() in (0, 1)


def test_low_temperature_sharpens_toward_the_argmax():
    """A low but nonzero temperature should make the argmax dominate, without making it certain."""
    logits = torch.tensor([[3.0, 2.9, 2.8]])
    generator = torch.Generator().manual_seed(0)

    picks = [
        PagedModel._sample_next_token(logits, temperature=0.05, top_k=None, top_p=None, generator=generator).item()
        for _ in range(20)
    ]

    assert picks.count(0) >= 15
