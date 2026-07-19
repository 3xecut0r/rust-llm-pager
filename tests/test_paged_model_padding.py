from __future__ import annotations

import pytest
import torch

from pager_hf.paged_model import PagedModel


def test_position_ids_all_ones_mask_is_plain_arange():
    mask = torch.ones(1, 5, dtype=torch.long)

    position_ids = PagedModel._position_ids_from_mask(mask)

    assert torch.equal(position_ids, torch.arange(5).unsqueeze(0))


def test_position_ids_left_padding_starts_at_zero_for_first_real_token():
    # 2 padding tokens, then 3 real tokens
    mask = torch.tensor([[0, 0, 1, 1, 1]])

    position_ids = PagedModel._position_ids_from_mask(mask)

    assert torch.equal(position_ids, torch.tensor([[0, 0, 0, 1, 2]]))


def test_position_ids_batch_rows_with_different_padding_amounts():
    mask = torch.tensor([[1, 1, 1, 1], [0, 1, 1, 1], [0, 0, 0, 1]])  # no padding  # 1 padding token  # 3 padding tokens

    position_ids = PagedModel._position_ids_from_mask(mask)

    expected = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 2], [0, 0, 0, 0]])
    assert torch.equal(position_ids, expected)


def test_validate_left_padded_accepts_all_ones():
    mask = torch.ones(2, 4, dtype=torch.long)
    PagedModel._validate_left_padded(mask)  # must not raise


def test_validate_left_padded_accepts_leading_zeros():
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    PagedModel._validate_left_padded(mask)  # must not raise


def test_validate_left_padded_rejects_right_padding():
    mask = torch.tensor([[1, 1, 1, 0]])
    with pytest.raises(NotImplementedError):
        PagedModel._validate_left_padded(mask)


def test_validate_left_padded_rejects_interleaved_mask():
    mask = torch.tensor([[0, 1, 0, 1]])
    with pytest.raises(NotImplementedError):
        PagedModel._validate_left_padded(mask)


def test_validate_left_padded_rejects_all_zero_row():
    mask = torch.tensor([[0, 0, 0, 0]])
    with pytest.raises(NotImplementedError):
        PagedModel._validate_left_padded(mask)
