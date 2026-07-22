from __future__ import annotations

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as tl_libdevice

# Inner tile size for the intra-kernel KV loop. Independent of tokens_per_block
# (the pager's own block size) -- this only controls how much K/V the kernel
# reads per iteration of its runtime-bound loop.
_BLOCK_KV = 64

# GQA query-head packing width. Every kernel launches one program per (batch
# row, KV head) pair, grid size batch * num_kv_heads. Each program packs the
# n_rep query heads that share this KV head into one [_M_TILE, head_dim] tile
# (padded with unused rows when n_rep < _M_TILE) so QK^T and P.V run as real
# tl.dot matmuls -- tensor-core eligible on Ampere+, plain FMA on Pascal,
# correct on both. This replaced an earlier one-program-per-query-head design
# using broadcast-multiply + tl.sum: that version never engages tensor cores
# on any hardware, and lost badly to PyTorch's cuBLAS-backed dense attention
# at head_dim=128 on real Ampere hardware (see README's "Important
# limitations" for the measured regression this fixes).
#
# q/m/l/acc/block_mass are laid out [batch, num_query_heads, ...]
# contiguously; a program's real rows are [head_offset, head_offset+n_rep)
# where head_offset = kv_head * n_rep (heads sharing a KV head are always
# contiguous). Padding rows (n_rep <= row < _M_TILE) index into a
# *neighboring* program's real data -- every load/store touching them is
# masked with row_mask, which Triton lowers to predicated instructions that
# never issue the actual memory transaction for masked-off lanes, so the
# out-of-range address is never dereferenced. Requires n_rep <= _M_TILE;
# every currently supported architecture (Qwen2, Llama, Mistral) has n_rep
# between 2 and 8, so this never triggers today, but a future architecture
# with wider GQA fan-out needs a bigger tile, hence the explicit guard.
#
# valid_ptr is a [batch, chunk_tokens] 0/1 mask excluding padding positions
# (left-padded rows in a batch>1 call) from attention -- without it, a
# shorter row's query would silently attend back to another row's padding
# columns, which are physically present in the gathered K/V but must never
# contribute. Combined with the existing out-of-range mask, not a replacement
# for it.
#
# active_ptr is a coarser, per-row [batch] 0/1 switch: when a row has *zero*
# real (valid) tokens anywhere in this call's chunk -- e.g. a short session
# batched cross-session (batched_decode.py) alongside a much longer one, in a
# KV group entirely past its own history -- the program skips the KV loop
# outright (loop trip count 0) instead of iterating the whole chunk only to
# have every position masked out by valid_ptr anyway. total_tokens itself is
# unchanged and still used for addressing (every row's K/V/valid tensors keep
# the same physical shape); only the trip count becomes per-row. A row with
# any real data in the chunk (including the one row per session whose own
# real/padding split falls inside this chunk) still runs the full loop,
# exactly as before -- valid_ptr already excludes the padding within it
# correctly, so there's nothing to gain by special-casing that row further.
_M_TILE = 16

# tl.dot requires its contraction dimension >= 16. head_dim is that dimension
# for QK^T (block_kv, the other dot's contraction dim, is always >= 16 by
# construction). Real models are always head_dim >= 64, but tiny test-only
# checkpoints (e.g. hf-internal-testing/tiny-random-MistralForCausalLM,
# head_dim=8) fall under that -- padding Q/K with zero columns up to
# _MIN_DOT_DIM leaves the dot product's value unchanged (zeros contribute
# nothing to the sum) while satisfying the shape constraint. A no-op for
# every currently supported real model, since qk_dim == head_dim whenever
# head_dim >= 16 already.
_MIN_DOT_DIM = 16


def _validate_n_rep(n_rep: int) -> None:
    if n_rep > _M_TILE:
        raise NotImplementedError(
            f"streaming attention's GQA query-head packing supports n_rep up to {_M_TILE}, got {n_rep}. "
            "This architecture's query/KV head ratio is wider than any currently supported model."
        )


@triton.jit
def _decode_attention_step_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_ptr,
    active_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    total_tokens,
    scale,
    softcap,
    n_rep: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_kv: tl.constexpr,
    m_tile: tl.constexpr,
    qk_dim: tl.constexpr,
):
    """
    query_len is always 1 (decode step), but the n_rep query heads sharing a
    KV head are packed into one [m_tile, head_dim] tile so QK^T/P.V run as
    tl.dot matmuls instead of per-head broadcast-multiply + tl.sum. Reads
    starting (m, l, acc) online-softmax state from m_ptr/l_ptr/acc_ptr and
    writes the updated state back, so a caller can fold in one KV group per
    launch across many launches -- only that group's K/V needs to be
    GPU-resident at once, not the whole context.

    scale multiplies the raw QK^T scores (the wrapper computes 1/sqrt(head_dim)
    when a caller doesn't override it -- e.g. Gemma2's query_pre_attn_scalar-based
    scale differs from that default). softcap, when > 0, applies Gemma2-style
    attn-logit softcapping (tanh(scores/softcap)*softcap) to the raw scores
    *before* the token_mask exclusion below -- matching Gemma2Attention.forward's
    own order (softcap on raw scores, then the causal/padding mask). softcap=0.0
    (the sentinel for "disabled", since Triton kernel args need a concrete value,
    not None) is a no-op, preserving every other architecture's exact behavior.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_kv_heads
    kv_head = pid % num_kv_heads
    head_offset = kv_head * n_rep

    row_offsets = tl.arange(0, m_tile)
    row_mask = row_offsets < n_rep
    full_pid = batch_idx * num_query_heads + head_offset + row_offsets

    dim_offsets = tl.arange(0, head_dim)
    qk_offsets = tl.arange(0, qk_dim)
    qk_col_mask = qk_offsets < head_dim

    q = tl.load(
        q_ptr + full_pid[:, None] * head_dim + qk_offsets[None, :],
        mask=row_mask[:, None] & qk_col_mask[None, :],
        other=0.0,
    )
    m = tl.load(m_ptr + full_pid, mask=row_mask, other=float("-inf"))
    l = tl.load(l_ptr + full_pid, mask=row_mask, other=0.0)
    acc = tl.load(acc_ptr + full_pid[:, None] * head_dim + dim_offsets[None, :], mask=row_mask[:, None], other=0.0)

    kv_head_base = batch_idx * num_kv_heads * total_tokens * head_dim + kv_head * total_tokens * head_dim
    valid_base = batch_idx * total_tokens
    row_active = tl.load(active_ptr + batch_idx)
    loop_bound = tl.where(row_active != 0, total_tokens, 0)

    for start in range(0, loop_bound, block_kv):
        offs = start + tl.arange(0, block_kv)
        token_mask = offs < total_tokens
        valid_chunk = tl.load(valid_ptr + valid_base + offs, mask=token_mask, other=0)
        token_mask = token_mask & (valid_chunk != 0)

        k_chunk = tl.load(
            k_ptr + kv_head_base + offs[:, None] * head_dim + qk_offsets[None, :],
            mask=token_mask[:, None] & qk_col_mask[None, :],
            other=0.0,
        )
        v_chunk = tl.load(
            v_ptr + kv_head_base + offs[:, None] * head_dim + dim_offsets[None, :], mask=token_mask[:, None], other=0.0
        )

        scores = tl.dot(q, tl.trans(k_chunk)) * scale
        if softcap > 0.0:
            scores = softcap * tl_libdevice.tanh(scores / softcap)
        scores = tl.where(token_mask[None, :], scores, float("-inf"))

        chunk_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, chunk_max)
        # m_new stays -inf when nothing valid has been seen yet through and
        # including this chunk (e.g. a session with zero real history blocks,
        # batched alongside sessions that do have some -- its every group is
        # entirely padding). m - m_new and scores - m_new are both -inf - -inf
        # (NaN) in exactly that case; alpha=0/p=0 there is the correct,
        # NaN-free no-op (acc/l are still their untouched zero-initial values).
        is_empty_so_far = m_new == float("-inf")
        alpha = tl.where(is_empty_so_far, 0.0, tl.exp(m - m_new))

        acc = acc * alpha[:, None]
        l = l * alpha

        p = tl.where(is_empty_so_far[:, None], 0.0, tl.exp(scores - m_new[:, None]))
        acc += tl.dot(p.to(v_chunk.dtype), v_chunk)
        l += tl.sum(p, axis=1)

        m = m_new

    tl.store(m_ptr + full_pid, m, mask=row_mask)
    tl.store(l_ptr + full_pid, l, mask=row_mask)
    tl.store(acc_ptr + full_pid[:, None] * head_dim + dim_offsets[None, :], acc, mask=row_mask[:, None])


@triton.jit
def _decode_attention_stats_kernel(
    q_ptr,
    k_ptr,
    valid_ptr,
    active_ptr,
    m_ptr,
    l_ptr,
    total_tokens,
    scale,
    softcap,
    n_rep: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_kv: tl.constexpr,
    m_tile: tl.constexpr,
    qk_dim: tl.constexpr,
):
    """Pass 1 of the mass-tracking path: fold one K chunk into running (m, l) only -- no V, no acc.
    scale/softcap: see _decode_attention_step_kernel's docstring."""
    pid = tl.program_id(0)
    batch_idx = pid // num_kv_heads
    kv_head = pid % num_kv_heads
    head_offset = kv_head * n_rep

    row_offsets = tl.arange(0, m_tile)
    row_mask = row_offsets < n_rep
    full_pid = batch_idx * num_query_heads + head_offset + row_offsets

    qk_offsets = tl.arange(0, qk_dim)
    qk_col_mask = qk_offsets < head_dim

    q = tl.load(
        q_ptr + full_pid[:, None] * head_dim + qk_offsets[None, :],
        mask=row_mask[:, None] & qk_col_mask[None, :],
        other=0.0,
    )
    m = tl.load(m_ptr + full_pid, mask=row_mask, other=float("-inf"))
    l = tl.load(l_ptr + full_pid, mask=row_mask, other=0.0)

    kv_head_base = batch_idx * num_kv_heads * total_tokens * head_dim + kv_head * total_tokens * head_dim
    valid_base = batch_idx * total_tokens
    row_active = tl.load(active_ptr + batch_idx)
    loop_bound = tl.where(row_active != 0, total_tokens, 0)

    for start in range(0, loop_bound, block_kv):
        offs = start + tl.arange(0, block_kv)
        token_mask = offs < total_tokens
        valid_chunk = tl.load(valid_ptr + valid_base + offs, mask=token_mask, other=0)
        token_mask = token_mask & (valid_chunk != 0)

        k_chunk = tl.load(
            k_ptr + kv_head_base + offs[:, None] * head_dim + qk_offsets[None, :],
            mask=token_mask[:, None] & qk_col_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(k_chunk)) * scale
        if softcap > 0.0:
            scores = softcap * tl_libdevice.tanh(scores / softcap)
        scores = tl.where(token_mask[None, :], scores, float("-inf"))

        chunk_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m, chunk_max)
        # See the matching comment in _decode_attention_step_kernel: guard the
        # nothing-valid-yet case (m_new stays -inf) to avoid -inf - -inf == NaN.
        is_empty_so_far = m_new == float("-inf")
        l = l * tl.where(is_empty_so_far, 0.0, tl.exp(m - m_new))
        l += tl.sum(tl.where(is_empty_so_far[:, None], 0.0, tl.exp(scores - m_new[:, None])), axis=1)
        m = m_new

    tl.store(m_ptr + full_pid, m, mask=row_mask)
    tl.store(l_ptr + full_pid, l, mask=row_mask)


@triton.jit
def _decode_attention_accumulate_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_ptr,
    active_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    block_mass_ptr,
    block_id_offset,
    total_tokens,
    scale,
    softcap,
    n_rep: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    tokens_per_block: tl.constexpr,
    max_blocks: tl.constexpr,
    m_tile: tl.constexpr,
    qk_dim: tl.constexpr,
):
    """
    Pass 2 of the mass-tracking path: (m, l) are already final (from pass 1)
    and read-only here. Accumulates acc (weighted V sum) and, per internal
    tokens_per_block-sized tile, the fraction of total attention mass that
    tile received -- one pager block's worth of score, in one clean number.
    A padded position still gets a (zero) mass entry rather than being
    dropped from the tile, so block ids stay aligned with the pager's own.

    scale/softcap: see _decode_attention_step_kernel's docstring.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_kv_heads
    kv_head = pid % num_kv_heads
    head_offset = kv_head * n_rep

    row_offsets = tl.arange(0, m_tile)
    row_mask = row_offsets < n_rep
    full_pid = batch_idx * num_query_heads + head_offset + row_offsets

    dim_offsets = tl.arange(0, head_dim)
    qk_offsets = tl.arange(0, qk_dim)
    qk_col_mask = qk_offsets < head_dim

    q = tl.load(
        q_ptr + full_pid[:, None] * head_dim + qk_offsets[None, :],
        mask=row_mask[:, None] & qk_col_mask[None, :],
        other=0.0,
    )
    m_final = tl.load(m_ptr + full_pid, mask=row_mask, other=float("-inf"))
    l_final = tl.load(l_ptr + full_pid, mask=row_mask, other=1.0)
    acc = tl.load(acc_ptr + full_pid[:, None] * head_dim + dim_offsets[None, :], mask=row_mask[:, None], other=0.0)

    kv_head_base = batch_idx * num_kv_heads * total_tokens * head_dim + kv_head * total_tokens * head_dim
    valid_base = batch_idx * total_tokens
    row_active = tl.load(active_ptr + batch_idx)
    num_local_blocks = tl.where(row_active != 0, (total_tokens + tokens_per_block - 1) // tokens_per_block, 0)

    for local_block in range(0, num_local_blocks):
        start = local_block * tokens_per_block
        offs = start + tl.arange(0, tokens_per_block)
        token_mask = offs < total_tokens
        valid_chunk = tl.load(valid_ptr + valid_base + offs, mask=token_mask, other=0)
        token_mask = token_mask & (valid_chunk != 0)

        k_chunk = tl.load(
            k_ptr + kv_head_base + offs[:, None] * head_dim + qk_offsets[None, :],
            mask=token_mask[:, None] & qk_col_mask[None, :],
            other=0.0,
        )
        v_chunk = tl.load(
            v_ptr + kv_head_base + offs[:, None] * head_dim + dim_offsets[None, :], mask=token_mask[:, None], other=0.0
        )

        scores = tl.dot(q, tl.trans(k_chunk)) * scale
        if softcap > 0.0:
            scores = softcap * tl_libdevice.tanh(scores / softcap)
        scores = tl.where(token_mask[None, :], scores, float("-inf"))

        p = tl.exp(scores - m_final[:, None])
        acc += tl.dot(p.to(v_chunk.dtype), v_chunk)

        block_mass = tl.sum(p, axis=1) / l_final
        tl.store(block_mass_ptr + full_pid * max_blocks + block_id_offset + local_block, block_mass, mask=row_mask)

    tl.store(acc_ptr + full_pid[:, None] * head_dim + dim_offsets[None, :], acc, mask=row_mask[:, None])


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
    scale,
    softcap,
    n_rep: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    block_kv: tl.constexpr,
    m_tile: tl.constexpr,
    qk_dim: tl.constexpr,
):
    """
    Same accumulation as pass 2, but for K/V that isn't a pager block (the
    tail) -- no block_mass write. m, l must already be final and are not
    updated further (the caller already folded this same K into them during
    pass 1, via _decode_attention_stats_kernel). scale/softcap: see
    _decode_attention_step_kernel's docstring.
    """
    pid = tl.program_id(0)
    batch_idx = pid // num_kv_heads
    kv_head = pid % num_kv_heads
    head_offset = kv_head * n_rep

    row_offsets = tl.arange(0, m_tile)
    row_mask = row_offsets < n_rep
    full_pid = batch_idx * num_query_heads + head_offset + row_offsets

    dim_offsets = tl.arange(0, head_dim)
    qk_offsets = tl.arange(0, qk_dim)
    qk_col_mask = qk_offsets < head_dim

    q = tl.load(
        q_ptr + full_pid[:, None] * head_dim + qk_offsets[None, :],
        mask=row_mask[:, None] & qk_col_mask[None, :],
        other=0.0,
    )
    m_final = tl.load(m_ptr + full_pid, mask=row_mask, other=float("-inf"))
    l_final = tl.load(l_ptr + full_pid, mask=row_mask, other=1.0)
    acc = tl.load(acc_ptr + full_pid[:, None] * head_dim + dim_offsets[None, :], mask=row_mask[:, None], other=0.0)

    kv_head_base = batch_idx * num_kv_heads * total_tokens * head_dim + kv_head * total_tokens * head_dim
    valid_base = batch_idx * total_tokens

    for start in range(0, total_tokens, block_kv):
        offs = start + tl.arange(0, block_kv)
        token_mask = offs < total_tokens
        valid_chunk = tl.load(valid_ptr + valid_base + offs, mask=token_mask, other=0)
        token_mask = token_mask & (valid_chunk != 0)

        k_chunk = tl.load(
            k_ptr + kv_head_base + offs[:, None] * head_dim + qk_offsets[None, :],
            mask=token_mask[:, None] & qk_col_mask[None, :],
            other=0.0,
        )
        v_chunk = tl.load(
            v_ptr + kv_head_base + offs[:, None] * head_dim + dim_offsets[None, :], mask=token_mask[:, None], other=0.0
        )

        scores = tl.dot(q, tl.trans(k_chunk)) * scale
        if softcap > 0.0:
            scores = softcap * tl_libdevice.tanh(scores / softcap)
        scores = tl.where(token_mask[None, :], scores, float("-inf"))

        p = tl.exp(scores - m_final[:, None])
        acc += tl.dot(p.to(v_chunk.dtype), v_chunk)

    tl.store(acc_ptr + full_pid[:, None] * head_dim + dim_offsets[None, :], acc, mask=row_mask[:, None])


def _default_valid(k: torch.Tensor) -> torch.Tensor:
    """All-valid mask (no padding excluded), used when a caller doesn't pass one -- e.g. synthetic/single-row tests."""
    batch_size, _, total_tokens, _ = k.shape
    return torch.ones((batch_size, total_tokens), dtype=torch.int32, device=k.device)


def _default_active(batch_size: int, device: torch.device) -> torch.Tensor:
    """All-active [batch] switch (no row skipped), used when a caller doesn't pass one -- preserves the pre-skip-optimization behavior exactly."""
    return torch.ones((batch_size,), dtype=torch.int32, device=device)


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
    active: torch.Tensor | None = None,
    scale: float | None = None,
    softcap: float | None = None,
) -> None:
    """
    Fold one KV chunk into the running (m, l, acc) state, in place.

    q: [batch, num_query_heads, head_dim]. k, v: [batch, num_kv_heads, chunk_tokens, head_dim].
    valid: optional [batch, chunk_tokens] 0/1 mask (1 = real token, 0 = padding); defaults to all-valid.
    active: optional [batch] 0/1 switch; a row with active=0 skips this chunk's KV loop
    entirely (must have zero valid tokens in it -- see the module-level active_ptr comment
    above _decode_attention_step_kernel); defaults to all-active (today's behavior, unchanged).
    scale: optional override for the QK^T score scale; defaults to 1/sqrt(head_dim) (every
    currently-supported architecture except Gemma2, which uses query_pre_attn_scalar instead).
    softcap: optional Gemma2-style attn-logit softcapping value; None/0.0 disables it (every
    other architecture).

    The kernel's pointer arithmetic assumes a tightly-packed layout; a
    non-contiguous slice of a larger tensor keeps the *original* tensor's
    stride between kv_heads, silently reading the wrong memory for
    kv_head > 0. contiguous() here is a correctness requirement.
    """
    batch_size, num_query_heads, head_dim = q.shape
    _, num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads
    _validate_n_rep(n_rep)
    qk_dim = max(head_dim, _MIN_DOT_DIM)
    if valid is None:
        valid = _default_valid(k)
    if active is None:
        active = _default_active(batch_size, k.device)
    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    _decode_attention_step_kernel[(batch_size * num_kv_heads,)](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        valid.contiguous(),
        active.contiguous(),
        m,
        l,
        acc,
        total_tokens,
        scale,
        softcap or 0.0,
        n_rep=n_rep,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_kv=_BLOCK_KV,
        m_tile=_M_TILE,
        qk_dim=qk_dim,
    )


def streaming_attention_stats_step(
    q: torch.Tensor,
    k: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    valid: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
    scale: float | None = None,
    softcap: float | None = None,
) -> None:
    """Pass 1: fold one KV chunk's K into running (m, l) only. q: [batch, num_query_heads, head_dim], k: [batch, num_kv_heads, chunk_tokens, head_dim].

    active: optional [batch] 0/1 switch; see streaming_attention_step's docstring. Defaults to all-active.
    scale/softcap: see streaming_attention_step's docstring.
    """
    batch_size, num_query_heads, head_dim = q.shape
    _, num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads
    _validate_n_rep(n_rep)
    qk_dim = max(head_dim, _MIN_DOT_DIM)
    if valid is None:
        valid = torch.ones((batch_size, total_tokens), dtype=torch.int32, device=k.device)
    if active is None:
        active = _default_active(batch_size, k.device)
    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    _decode_attention_stats_kernel[(batch_size * num_kv_heads,)](
        q.contiguous(),
        k.contiguous(),
        valid.contiguous(),
        active.contiguous(),
        m,
        l,
        total_tokens,
        scale,
        softcap or 0.0,
        n_rep=n_rep,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_kv=_BLOCK_KV,
        m_tile=_M_TILE,
        qk_dim=qk_dim,
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
    active: torch.Tensor | None = None,
    scale: float | None = None,
    softcap: float | None = None,
) -> None:
    """Pass 2: fold one pager-block-aligned KV group into acc, and write its per-block attention mass.

    active: optional [batch] 0/1 switch; see streaming_attention_step's docstring. A skipped row's
    block_mass entries for this group are left untouched -- correct as long as block_mass was
    zero-initialized by the caller (matching what the masked-out computation would have produced
    anyway), which batched_decode_step already does. Defaults to all-active.
    scale/softcap: see streaming_attention_step's docstring.
    """
    batch_size, num_query_heads, head_dim = q.shape
    _, num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads
    _validate_n_rep(n_rep)
    qk_dim = max(head_dim, _MIN_DOT_DIM)
    max_blocks = block_mass.shape[2]
    if valid is None:
        valid = _default_valid(k)
    if active is None:
        active = _default_active(batch_size, k.device)
    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    _decode_attention_accumulate_kernel[(batch_size * num_kv_heads,)](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        valid.contiguous(),
        active.contiguous(),
        m,
        l,
        acc,
        block_mass,
        block_id_offset,
        total_tokens,
        scale,
        softcap or 0.0,
        n_rep=n_rep,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_blocks=max_blocks,
        m_tile=_M_TILE,
        qk_dim=qk_dim,
    )


def streaming_attention_fold_fixed_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    m: torch.Tensor,
    l: torch.Tensor,
    acc: torch.Tensor,
    valid: torch.Tensor | None = None,
    scale: float | None = None,
    softcap: float | None = None,
) -> None:
    """Pass 2 fold for a non-block chunk (the tail): accumulate acc only, no mass write.
    scale/softcap: see streaming_attention_step's docstring."""
    batch_size, num_query_heads, head_dim = q.shape
    _, num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads
    _validate_n_rep(n_rep)
    qk_dim = max(head_dim, _MIN_DOT_DIM)
    if valid is None:
        valid = _default_valid(k)
    if scale is None:
        scale = 1.0 / (head_dim**0.5)

    _decode_attention_fold_fixed_kernel[(batch_size * num_kv_heads,)](
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        valid.contiguous(),
        m,
        l,
        acc,
        total_tokens,
        scale,
        softcap or 0.0,
        n_rep=n_rep,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        block_kv=_BLOCK_KV,
        m_tile=_M_TILE,
        qk_dim=qk_dim,
    )


def streaming_attention_finalize(acc: torch.Tensor, l: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Turn accumulated (acc, l) state into the final attention output, once every chunk has been folded in."""
    return (acc / l.unsqueeze(-1)).to(dtype)
