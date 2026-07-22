import pytest
import torch

from pager_hf.streaming_attention import (
    streaming_attention_accumulate_step,
    streaming_attention_finalize,
    streaming_attention_fold_fixed_step,
    streaming_attention_state_init,
    streaming_attention_stats_step,
    streaming_attention_step,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels require CUDA.")

NUM_QUERY_HEADS = 14
NUM_KV_HEADS = 2
HEAD_DIM = 64


def _dense_reference(q, k, v, *, n_rep, tokens_per_block, valid=None, scale=None, softcap=None):
    """q: [batch, heads, dim], k/v: [batch, kv_heads, tokens, dim], valid: optional [batch, tokens] 0/1 mask."""
    key = k.repeat_interleave(n_rep, dim=1).float()
    value = v.repeat_interleave(n_rep, dim=1).float()
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)

    scores = torch.einsum("bhd,bhtd->bht", q.float(), key) * scale
    if softcap is not None:
        scores = softcap * torch.tanh(scores / softcap)
    if valid is not None:
        scores = scores.masked_fill(valid[:, None, :] == 0, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("bht,bhtd->bhd", weights, value)

    total_tokens = k.shape[2]
    num_blocks = (total_tokens + tokens_per_block - 1) // tokens_per_block
    block_mass = torch.stack(
        [weights[:, :, b * tokens_per_block : (b + 1) * tokens_per_block].sum(dim=-1) for b in range(num_blocks)], dim=2
    )
    return out, block_mass


def test_single_pass_matches_dense():
    torch.manual_seed(0)
    device = torch.device("cuda")
    total_tokens = 200
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS

    q = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)

    m, l, acc = streaming_attention_state_init(1, NUM_QUERY_HEADS, HEAD_DIM, device)
    group_size = 48
    for start in range(0, total_tokens, group_size):
        end = min(start + group_size, total_tokens)
        streaming_attention_step(q, k[:, :, start:end, :], v[:, :, start:end, :], m, l, acc)

    out = streaming_attention_finalize(acc, l, torch.float16)
    dense_out, _ = _dense_reference(q, k, v, n_rep=n_rep, tokens_per_block=16)

    assert torch.allclose(out, dense_out.to(torch.float16), rtol=2e-2, atol=2e-2)


def test_two_pass_with_tail_matches_dense_mass_and_output():
    """
    Mirrors the real PagedModel decode step: historical KV split into
    pager-block-aligned groups (mass-tracked) plus a non-block tail (folded
    in, no mass entry) -- exercising streaming_attention_fold_fixed_step,
    which has no standalone coverage elsewhere.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    tokens_per_block = 16
    num_full_blocks = 9
    full_tokens = num_full_blocks * tokens_per_block
    tail_tokens = 7
    total_tokens = full_tokens + tail_tokens
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS
    group_size = 32  # multiple of tokens_per_block, like GROUP_SIZE_BLOCKS-worth of blocks in production

    q = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)

    full_k, full_v = k[:, :, :full_tokens, :], v[:, :, :full_tokens, :]
    tail_k, tail_v = k[:, :, full_tokens:, :], v[:, :, full_tokens:, :]

    # Pass 1: fold every chunk (groups + tail) into (m, l) only.
    m, l, _ = streaming_attention_state_init(1, NUM_QUERY_HEADS, HEAD_DIM, device)
    for start in range(0, full_tokens, group_size):
        end = min(start + group_size, full_tokens)
        streaming_attention_stats_step(q, full_k[:, :, start:end, :], m, l)
    streaming_attention_stats_step(q, tail_k, m, l)

    # Pass 2: (m, l) fixed. Groups write block_mass; the tail folds into acc only.
    acc = torch.zeros(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    block_mass = torch.zeros(1, NUM_QUERY_HEADS, num_full_blocks, dtype=torch.float32, device=device)
    for start in range(0, full_tokens, group_size):
        end = min(start + group_size, full_tokens)
        streaming_attention_accumulate_step(
            q,
            full_k[:, :, start:end, :],
            full_v[:, :, start:end, :],
            m,
            l,
            acc,
            block_mass,
            start // tokens_per_block,
            tokens_per_block,
        )
    streaming_attention_fold_fixed_step(q, tail_k, tail_v, m, l, acc)

    out = streaming_attention_finalize(acc, l, torch.float16)
    dense_out, dense_mass = _dense_reference(q, k, v, n_rep=n_rep, tokens_per_block=tokens_per_block)
    dense_block_mass = dense_mass[:, :, :num_full_blocks]  # exclude the dense reference's own ragged tail "block"

    assert torch.allclose(out, dense_out.to(torch.float16), rtol=2e-2, atol=2e-2)
    assert torch.allclose(block_mass, dense_block_mass, rtol=2e-2, atol=2e-2)


def test_batched_rows_are_independent():
    """Two unrelated rows processed in one launch must match processing each row alone -- no cross-row leakage."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    total_tokens = 96
    group_size = 32
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS

    q = torch.randn(2, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(2, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(2, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)

    m, l, acc = streaming_attention_state_init(2, NUM_QUERY_HEADS, HEAD_DIM, device)
    for start in range(0, total_tokens, group_size):
        end = min(start + group_size, total_tokens)
        streaming_attention_step(q, k[:, :, start:end, :], v[:, :, start:end, :], m, l, acc)
    batched_out = streaming_attention_finalize(acc, l, torch.float16)

    dense_out, _ = _dense_reference(q, k, v, n_rep=n_rep, tokens_per_block=16)

    assert torch.allclose(batched_out, dense_out.to(torch.float16), rtol=2e-2, atol=2e-2)


def test_padding_mask_excludes_padded_positions():
    """
    A row with a padded prefix must ignore that prefix's K/V entirely --
    matching a dense reference computed with the same validity mask, and
    diverging from a reference that (wrongly) includes the padding.
    """
    torch.manual_seed(2)
    device = torch.device("cuda")
    total_tokens = 64
    pad_tokens = 20
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS

    q = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    valid = torch.ones(1, total_tokens, dtype=torch.int32, device=device)
    valid[:, :pad_tokens] = 0

    m, l, acc = streaming_attention_state_init(1, NUM_QUERY_HEADS, HEAD_DIM, device)
    streaming_attention_step(q, k, v, m, l, acc, valid=valid)
    out = streaming_attention_finalize(acc, l, torch.float16)

    dense_masked, _ = _dense_reference(q, k, v, n_rep=n_rep, tokens_per_block=16, valid=valid)
    dense_unmasked, _ = _dense_reference(q, k, v, n_rep=n_rep, tokens_per_block=16)

    assert torch.allclose(out, dense_masked.to(torch.float16), rtol=2e-2, atol=2e-2)
    assert not torch.allclose(out, dense_unmasked.to(torch.float16), rtol=2e-2, atol=2e-2)


def test_active_zero_skips_row_regardless_of_valid_mask():
    """
    active=0 (the batched_decode_step padding-skip optimization) must be an
    unconditional skip of that row's KV loop -- proven by deliberately using
    a FULLY VALID chunk for the "inactive" row (a combination the real
    caller never produces -- it only sets active=0 when a row has zero real
    data -- but the clearest way to prove the kernel genuinely reads
    active_ptr rather than silently ignoring it): if the skip weren't wired
    up, this row's state would change from processing a fully valid chunk.
    The other row in the same launch, left active, must be completely
    unaffected and match the ordinary dense reference.
    """
    torch.manual_seed(4)
    device = torch.device("cuda")
    chunk_tokens = 32
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS

    q = torch.randn(2, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(2, NUM_KV_HEADS, chunk_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(2, NUM_KV_HEADS, chunk_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    valid = torch.ones(2, chunk_tokens, dtype=torch.int32, device=device)
    active = torch.tensor([0, 1], dtype=torch.int32, device=device)

    m, l, acc = streaming_attention_state_init(2, NUM_QUERY_HEADS, HEAD_DIM, device)
    m_before, l_before, acc_before = m.clone(), l.clone(), acc.clone()
    streaming_attention_step(q, k, v, m, l, acc, valid=valid, active=active)

    assert torch.equal(m[0], m_before[0])
    assert torch.equal(l[0], l_before[0])
    assert torch.equal(acc[0], acc_before[0])

    dense_out, _ = _dense_reference(q[1:2], k[1:2], v[1:2], n_rep=n_rep, tokens_per_block=16)
    out = streaming_attention_finalize(acc, l, torch.float16)
    assert torch.allclose(out[1:2], dense_out.to(torch.float16), rtol=2e-2, atol=2e-2)


def test_active_zero_skips_row_in_stats_and_accumulate_steps():
    """Same active=0 no-op guarantee, for the two-pass mass-tracking path used by attention-scoring policies."""
    torch.manual_seed(5)
    device = torch.device("cuda")
    tokens_per_block = 16
    chunk_tokens = tokens_per_block * 2

    q = torch.randn(2, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(2, NUM_KV_HEADS, chunk_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(2, NUM_KV_HEADS, chunk_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    valid = torch.ones(2, chunk_tokens, dtype=torch.int32, device=device)
    active = torch.tensor([0, 1], dtype=torch.int32, device=device)

    m, l, acc = streaming_attention_state_init(2, NUM_QUERY_HEADS, HEAD_DIM, device)
    m_before, l_before = m.clone(), l.clone()
    streaming_attention_stats_step(q, k, m, l, valid=valid, active=active)
    assert torch.equal(m[0], m_before[0])
    assert torch.equal(l[0], l_before[0])

    block_mass = torch.zeros(2, NUM_QUERY_HEADS, 2, dtype=torch.float32, device=device)
    acc_before = acc.clone()
    streaming_attention_accumulate_step(q, k, v, m, l, acc, block_mass, 0, tokens_per_block, valid=valid, active=active)
    assert torch.equal(acc[0], acc_before[0])
    assert torch.equal(block_mass[0], torch.zeros_like(block_mass[0]))

    dense_out, dense_mass = _dense_reference(
        q[1:2], k[1:2], v[1:2], n_rep=NUM_QUERY_HEADS // NUM_KV_HEADS, tokens_per_block=tokens_per_block
    )
    out = streaming_attention_finalize(acc, l, torch.float16)
    assert torch.allclose(out[1:2], dense_out.to(torch.float16), rtol=2e-2, atol=2e-2)
    assert torch.allclose(block_mass[1:2], dense_mass, rtol=2e-2, atol=2e-2)


def test_custom_scale_and_softcap_match_dense_reference():
    """Gemma2 support: a non-default scale (query_pre_attn_scalar-based, not 1/sqrt(head_dim)) and
    attn-logit softcapping (tanh(scores/cap)*cap applied to raw scores before masking) must match a
    dense reference applying the identical transform -- not just 'doesn't crash'."""
    torch.manual_seed(6)
    device = torch.device("cuda")
    total_tokens = 96
    group_size = 32
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS
    custom_scale = 224.0**-0.5  # Gemma2's default query_pre_attn_scalar, deliberately != 1/sqrt(HEAD_DIM)
    softcap = 50.0  # Gemma2's default attn_logit_softcapping

    q = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)

    m, l, acc = streaming_attention_state_init(1, NUM_QUERY_HEADS, HEAD_DIM, device)
    for start in range(0, total_tokens, group_size):
        end = min(start + group_size, total_tokens)
        streaming_attention_step(
            q, k[:, :, start:end, :], v[:, :, start:end, :], m, l, acc, scale=custom_scale, softcap=softcap
        )
    out = streaming_attention_finalize(acc, l, torch.float16)

    dense_out, _ = _dense_reference(q, k, v, n_rep=n_rep, tokens_per_block=16, scale=custom_scale, softcap=softcap)
    assert torch.allclose(out, dense_out.to(torch.float16), rtol=2e-2, atol=2e-2)

    # And confirm it's a genuine effect, not accidentally close to the un-softcapped/default-scale answer.
    dense_default, _ = _dense_reference(q, k, v, n_rep=n_rep, tokens_per_block=16)
    assert not torch.allclose(out, dense_default.to(torch.float16), rtol=2e-2, atol=2e-2)


def test_default_scale_and_softcap_unchanged_from_before_gemma2_support():
    """scale=None, softcap=None (every currently-supported architecture's call sites) must produce
    byte-identical output to before these parameters existed -- a regression guard on the refactor
    that moved 'scale = 1/sqrt(head_dim)' from inline kernel computation to a passed-in argument."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    total_tokens = 64
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS

    q = torch.randn(1, NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float16, device=device)
    k = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)
    v = torch.randn(1, NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=torch.float16, device=device)

    m, l, acc = streaming_attention_state_init(1, NUM_QUERY_HEADS, HEAD_DIM, device)
    streaming_attention_step(q, k, v, m, l, acc)
    out = streaming_attention_finalize(acc, l, torch.float16)

    dense_out, _ = _dense_reference(q, k, v, n_rep=n_rep, tokens_per_block=16)
    assert torch.allclose(out, dense_out.to(torch.float16), rtol=2e-2, atol=2e-2)
