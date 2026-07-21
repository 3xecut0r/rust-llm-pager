from __future__ import annotations

import torch
import triton
import triton.language as tl

# Inner tile size for the intra-kernel KV loop. Independent of tokens_per_block
# (the pager's own block size) -- this only controls how much K/V the kernel
# reads per iteration of its runtime-bound loop.
_BLOCK_KV = 64

# Every kernel launches one program per (batch row, query head) pair, grid
# size batch * num_query_heads. q/m/l/acc/block_mass are laid out
# [batch, num_query_heads, ...] contiguously, so pid already addresses them
# directly (pid * head_dim, pid, pid * max_blocks, ...) without decomposing
# it -- only K/V/valid addressing needs batch_idx/kv_head split out, since
# they're [batch, num_kv_heads or 1, tokens, ...] (a different per-batch stride).
#
# valid_ptr is a [batch, chunk_tokens] 0/1 mask excluding padding positions
# (left-padded rows in a batch>1 call) from attention -- without it, a
# shorter row's query would silently attend back to another row's padding
# columns, which are physically present in the gathered K/V but must never
# contribute. Combined with the existing out-of-range mask, not a replacement
# for it.


@triton.jit
def _decode_attention_step_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    total_tokens,
    n_rep: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_kv: tl.constexpr,
):
    """
    query_len is always 1 (decode step), so QK^T and P.V are matrix-vector
    products, computed as broadcast-multiply + tl.sum (flash-decoding
    pattern) rather than tl.dot (which wants an M>=16 tile). Reads starting
    (m, l, acc) online-softmax state from m_ptr/l_ptr/acc_ptr and writes the
    updated state back, so a caller can fold in one KV group per launch
    across many launches -- only that group's K/V needs to be GPU-resident
    at once, not the whole context.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_query_heads
    head_idx = pid % num_query_heads
    kv_head = head_idx // n_rep

    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + pid * head_dim + dim_offsets).to(tl.float32)

    scale = 1.0 / (head_dim**0.5)

    m = tl.load(m_ptr + pid)
    l = tl.load(l_ptr + pid)
    acc = tl.load(acc_ptr + pid * head_dim + dim_offsets)

    kv_head_base = batch_idx * num_kv_heads * total_tokens * head_dim + kv_head * total_tokens * head_dim
    valid_base = batch_idx * total_tokens

    for start in range(0, total_tokens, block_kv):
        offs = start + tl.arange(0, block_kv)
        token_mask = offs < total_tokens
        valid_chunk = tl.load(valid_ptr + valid_base + offs, mask=token_mask, other=0)
        token_mask = token_mask & (valid_chunk != 0)

        kv_ptrs = kv_head_base + offs[:, None] * head_dim + dim_offsets[None, :]
        k_chunk = tl.load(k_ptr + kv_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        v_chunk = tl.load(v_ptr + kv_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k_chunk, axis=1) * scale
        scores = tl.where(token_mask, scores, float("-inf"))

        chunk_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m, chunk_max)
        alpha = tl.exp(m - m_new)

        acc = acc * alpha
        l = l * alpha

        p = tl.exp(scores - m_new)
        acc += tl.sum(p[:, None] * v_chunk, axis=0)
        l += tl.sum(p, axis=0)

        m = m_new

    tl.store(m_ptr + pid, m)
    tl.store(l_ptr + pid, l)
    tl.store(acc_ptr + pid * head_dim + dim_offsets, acc)


@triton.jit
def _decode_attention_stats_kernel(
    q_ptr,
    k_ptr,
    valid_ptr,
    m_ptr,
    l_ptr,
    total_tokens,
    n_rep: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_kv: tl.constexpr,
):
    """Pass 1 of the mass-tracking path: fold one K chunk into running (m, l) only -- no V, no acc."""
    pid = tl.program_id(0)
    batch_idx = pid // num_query_heads
    head_idx = pid % num_query_heads
    kv_head = head_idx // n_rep

    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + pid * head_dim + dim_offsets).to(tl.float32)

    scale = 1.0 / (head_dim**0.5)

    m = tl.load(m_ptr + pid)
    l = tl.load(l_ptr + pid)

    kv_head_base = batch_idx * num_kv_heads * total_tokens * head_dim + kv_head * total_tokens * head_dim
    valid_base = batch_idx * total_tokens

    for start in range(0, total_tokens, block_kv):
        offs = start + tl.arange(0, block_kv)
        token_mask = offs < total_tokens
        valid_chunk = tl.load(valid_ptr + valid_base + offs, mask=token_mask, other=0)
        token_mask = token_mask & (valid_chunk != 0)

        k_chunk = tl.load(
            k_ptr + kv_head_base + offs[:, None] * head_dim + dim_offsets[None, :], mask=token_mask[:, None], other=0.0
        ).to(tl.float32)

        scores = tl.sum(q[None, :] * k_chunk, axis=1) * scale
        scores = tl.where(token_mask, scores, float("-inf"))

        chunk_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m, chunk_max)
        l = l * tl.exp(m - m_new)
        l += tl.sum(tl.exp(scores - m_new), axis=0)
        m = m_new

    tl.store(m_ptr + pid, m)
    tl.store(l_ptr + pid, l)


@triton.jit
def _decode_attention_accumulate_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    block_mass_ptr,
    block_id_offset,
    total_tokens,
    n_rep: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    tokens_per_block: tl.constexpr,
    max_blocks: tl.constexpr,
):
    """
    Pass 2 of the mass-tracking path: (m, l) are already final (from pass 1)
    and read-only here. Accumulates acc (weighted V sum) and, per internal
    tokens_per_block-sized tile, the fraction of total attention mass that
    tile received -- one pager block's worth of score, in one clean number.
    A padded position still gets a (zero) mass entry rather than being
    dropped from the tile, so block ids stay aligned with the pager's own.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_query_heads
    head_idx = pid % num_query_heads
    kv_head = head_idx // n_rep

    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + pid * head_dim + dim_offsets).to(tl.float32)

    scale = 1.0 / (head_dim**0.5)

    m_final = tl.load(m_ptr + pid)
    l_final = tl.load(l_ptr + pid)
    acc = tl.load(acc_ptr + pid * head_dim + dim_offsets)

    kv_head_base = batch_idx * num_kv_heads * total_tokens * head_dim + kv_head * total_tokens * head_dim
    valid_base = batch_idx * total_tokens
    num_local_blocks = (total_tokens + tokens_per_block - 1) // tokens_per_block

    for local_block in range(0, num_local_blocks):
        start = local_block * tokens_per_block
        offs = start + tl.arange(0, tokens_per_block)
        token_mask = offs < total_tokens
        valid_chunk = tl.load(valid_ptr + valid_base + offs, mask=token_mask, other=0)
        token_mask = token_mask & (valid_chunk != 0)

        kv_ptrs = kv_head_base + offs[:, None] * head_dim + dim_offsets[None, :]
        k_chunk = tl.load(k_ptr + kv_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        v_chunk = tl.load(v_ptr + kv_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k_chunk, axis=1) * scale
        scores = tl.where(token_mask, scores, float("-inf"))

        p = tl.exp(scores - m_final)
        acc += tl.sum(p[:, None] * v_chunk, axis=0)

        block_mass = tl.sum(p, axis=0) / l_final
        tl.store(block_mass_ptr + pid * max_blocks + block_id_offset + local_block, block_mass)

    tl.store(acc_ptr + pid * head_dim + dim_offsets, acc)


@triton.jit
def _decode_attention_fold_fixed_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    total_tokens,
    n_rep: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_kv: tl.constexpr,
):
    """
    Same accumulation as pass 2, but for K/V that isn't a pager block (the
    tail) -- no block_mass write. m, l must already be final and are not
    updated further (the caller already folded this same K into them during
    pass 1, via _decode_attention_stats_kernel).
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_query_heads
    head_idx = pid % num_query_heads
    kv_head = head_idx // n_rep

    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + pid * head_dim + dim_offsets).to(tl.float32)

    scale = 1.0 / (head_dim**0.5)

    m_final = tl.load(m_ptr + pid)
    l_final = tl.load(l_ptr + pid)
    acc = tl.load(acc_ptr + pid * head_dim + dim_offsets)

    kv_head_base = batch_idx * num_kv_heads * total_tokens * head_dim + kv_head * total_tokens * head_dim
    valid_base = batch_idx * total_tokens

    for start in range(0, total_tokens, block_kv):
        offs = start + tl.arange(0, block_kv)
        token_mask = offs < total_tokens
        valid_chunk = tl.load(valid_ptr + valid_base + offs, mask=token_mask, other=0)
        token_mask = token_mask & (valid_chunk != 0)

        kv_ptrs = kv_head_base + offs[:, None] * head_dim + dim_offsets[None, :]
        k_chunk = tl.load(k_ptr + kv_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)
        v_chunk = tl.load(v_ptr + kv_ptrs, mask=token_mask[:, None], other=0.0).to(tl.float32)

        scores = tl.sum(q[None, :] * k_chunk, axis=1) * scale
        scores = tl.where(token_mask, scores, float("-inf"))

        p = tl.exp(scores - m_final)
        acc += tl.sum(p[:, None] * v_chunk, axis=0)

    tl.store(acc_ptr + pid * head_dim + dim_offsets, acc)


def _default_valid(k: torch.Tensor) -> torch.Tensor:
    """All-valid mask (no padding excluded), used when a caller doesn't pass one -- e.g. synthetic/single-row tests."""
    batch_size, _, total_tokens, _ = k.shape
    return torch.ones((batch_size, total_tokens), dtype=torch.int32, device=k.device)


def streaming_attention_state_init(batch_size: int, num_query_heads: int, head_dim: int, device: torch.device):
    """Fresh online-softmax accumulator state, before any KV chunk has been folded in."""
    m = torch.full((batch_size, num_query_heads), float("-inf"), dtype=torch.float32, device=device)
    l = torch.zeros((batch_size, num_query_heads), dtype=torch.float32, device=device)
    acc = torch.zeros((batch_size, num_query_heads, head_dim), dtype=torch.float32, device=device)
    return m, l, acc


def streaming_attention_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    acc: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> None:
    """
    Fold one KV chunk into the running (m, l, acc) state, in place.

    q: [batch, num_query_heads, head_dim]. k, v: [batch, num_kv_heads, chunk_tokens, head_dim].
    valid: optional [batch, chunk_tokens] 0/1 mask (1 = real token, 0 = padding); defaults to all-valid.

    The kernel's pointer arithmetic assumes a tightly-packed layout; a
    non-contiguous slice of a larger tensor keeps the *original* tensor's
    stride between kv_heads, silently reading the wrong memory for
    kv_head > 0. contiguous() here is a correctness requirement.
    """
    batch_size, num_query_heads, head_dim = q.shape
    _, num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads
    if valid is None:
        valid = _default_valid(k)

    _decode_attention_step_kernel[(batch_size * num_query_heads,)](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        valid.contiguous(),
        m,
        l,
        acc,
        total_tokens,
        n_rep=n_rep,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_kv=_BLOCK_KV,
    )


def streaming_attention_stats_step(
    q: torch.Tensor, k: torch.Tensor, m: torch.Tensor, l: torch.Tensor, valid: torch.Tensor | None = None
) -> None:
    """Pass 1: fold one KV chunk's K into running (m, l) only. q: [batch, num_query_heads, head_dim], k: [batch, num_kv_heads, chunk_tokens, head_dim]."""
    batch_size, num_query_heads, head_dim = q.shape
    _, num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads
    if valid is None:
        valid = torch.ones((batch_size, total_tokens), dtype=torch.int32, device=k.device)

    _decode_attention_stats_kernel[(batch_size * num_query_heads,)](
        q.contiguous(),
        k.contiguous(),
        valid.contiguous(),
        m,
        l,
        total_tokens,
        n_rep=n_rep,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_kv=_BLOCK_KV,
    )


def streaming_attention_accumulate_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    acc: torch.Tensor,
    block_mass: torch.Tensor,
    block_id_offset: int,
    tokens_per_block: int,
    valid: torch.Tensor | None = None,
) -> None:
    """Pass 2: fold one pager-block-aligned KV group into acc, and write its per-block attention mass."""
    batch_size, num_query_heads, head_dim = q.shape
    _, num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads
    max_blocks = block_mass.shape[2]
    if valid is None:
        valid = _default_valid(k)

    _decode_attention_accumulate_kernel[(batch_size * num_query_heads,)](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        valid.contiguous(),
        m,
        l,
        acc,
        block_mass,
        block_id_offset,
        total_tokens,
        n_rep=n_rep,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_blocks=max_blocks,
    )


def streaming_attention_fold_fixed_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    acc: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> None:
    """Pass 2 fold for a non-block chunk (the tail): accumulate acc only, no mass write."""
    batch_size, num_query_heads, head_dim = q.shape
    _, num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads
    if valid is None:
        valid = _default_valid(k)

    _decode_attention_fold_fixed_kernel[(batch_size * num_query_heads,)](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        valid.contiguous(),
        m,
        l,
        acc,
        total_tokens,
        n_rep=n_rep,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_kv=_BLOCK_KV,
    )


def streaming_attention_finalize(acc: torch.Tensor, l: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Turn accumulated (acc, l) state into the final attention output, once every chunk has been folded in."""
    return (acc / l.unsqueeze(-1)).to(dtype)
