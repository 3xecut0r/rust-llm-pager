from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

HEAD_DIM = 64
NUM_QUERY_HEADS = 14
NUM_KV_HEADS = 2
BLOCK_KV = 64


@triton.jit
def decode_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, total_tokens, n_rep: tl.constexpr, head_dim: tl.constexpr, block_kv: tl.constexpr
):
    """
    One program per query head. query_len is always 1 (this project's decode
    step), so QK^T and P.V are matrix-vector products -- computed as
    broadcast-multiply + tl.sum (the flash-decoding pattern), not tl.dot,
    which wants an M>=16 tile. Loops over KV in block_kv-sized chunks with a
    runtime bound (total_tokens isn't known at compile time), carrying
    online-softmax state (m, l, acc) in fp32 across the loop.
    """
    pid = tl.program_id(0)
    kv_head = pid // n_rep

    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + pid * head_dim + dim_offsets).to(tl.float32)

    scale = 1.0 / (head_dim**0.5)

    m = float("-inf")
    l = 0.0
    acc = tl.zeros((head_dim,), dtype=tl.float32)

    kv_head_base = kv_head * total_tokens * head_dim

    for start in range(0, total_tokens, block_kv):
        offs = start + tl.arange(0, block_kv)
        token_mask = offs < total_tokens

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

    out = acc / l
    tl.store(out_ptr + pid * head_dim + dim_offsets, out.to(out_ptr.dtype.element_ty))


def triton_decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    q: [num_query_heads, head_dim]
    k, v: [num_kv_heads, total_tokens, head_dim], contiguous
    Returns: [num_query_heads, head_dim], same dtype as q.
    """
    num_query_heads, head_dim = q.shape
    num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads

    out = torch.empty_like(q)

    decode_attention_kernel[(num_query_heads,)](
        q, k.contiguous(), v.contiguous(), out, total_tokens, n_rep=n_rep, head_dim=head_dim, block_kv=BLOCK_KV
    )
    return out


@triton.jit
def decode_attention_step_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    total_tokens,
    n_rep: tl.constexpr,
    head_dim: tl.constexpr,
    block_kv: tl.constexpr,
):
    """
    Same online-softmax math as decode_attention_kernel, but reads its
    starting (m, l, acc) state from m_ptr/l_ptr/acc_ptr and writes the
    updated state back to the same buffers, instead of starting fresh and
    finalizing internally. Lets a caller process a KV context in several
    small groups across several kernel launches -- each launch only ever
    needs that one group's K/V resident on GPU, not the whole context --
    continuing the same running accumulation from where the last call left off.
    """
    pid = tl.program_id(0)
    kv_head = pid // n_rep

    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + pid * head_dim + dim_offsets).to(tl.float32)

    scale = 1.0 / (head_dim**0.5)

    m = tl.load(m_ptr + pid)
    l = tl.load(l_ptr + pid)
    acc = tl.load(acc_ptr + pid * head_dim + dim_offsets)

    kv_head_base = kv_head * total_tokens * head_dim

    for start in range(0, total_tokens, block_kv):
        offs = start + tl.arange(0, block_kv)
        token_mask = offs < total_tokens

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


def streaming_attention_state_init(num_query_heads: int, head_dim: int, device: torch.device):
    """Fresh online-softmax accumulator state, before any group has been processed."""
    m = torch.full((num_query_heads,), float("-inf"), dtype=torch.float32, device=device)
    l = torch.zeros((num_query_heads,), dtype=torch.float32, device=device)
    acc = torch.zeros((num_query_heads, head_dim), dtype=torch.float32, device=device)
    return m, l, acc


def streaming_attention_step(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, m: torch.Tensor, l: torch.Tensor, acc: torch.Tensor
) -> None:
    """
    Fold one group's K/V into the running (m, l, acc) state, in place.

    The kernel's own pointer arithmetic assumes a tightly-packed
    [kv_heads, group_tokens, head_dim] layout; a non-contiguous slice of a
    larger tensor keeps the *original* tensor's stride between kv_heads,
    which silently reads the wrong memory for kv_head > 0. contiguous() is
    a real correctness requirement here, not just a performance nicety.
    """
    num_query_heads, head_dim = q.shape
    num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads

    decode_attention_step_kernel[(num_query_heads,)](
        q, k.contiguous(), v.contiguous(), m, l, acc, total_tokens, n_rep=n_rep, head_dim=head_dim, block_kv=BLOCK_KV
    )


def streaming_attention_finalize(acc: torch.Tensor, l: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Turn the accumulated (acc, l) state into the final attention output, once every group has been folded in."""
    return (acc / l.unsqueeze(-1)).to(dtype)


@triton.jit
def decode_attention_stats_kernel(
    q_ptr, k_ptr, m_ptr, l_ptr, total_tokens, n_rep: tl.constexpr, head_dim: tl.constexpr, block_kv: tl.constexpr
):
    """
    Pass 1 of the attention-scoring path: fold one group's K into the
    running (m, l) online-softmax statistics only -- no V, no output
    accumulator. Used to find the true final (m, l) across every block
    before pass 2 computes per-block attention mass against them, so pass 2
    never needs to rescale an already-written mass value.
    """
    pid = tl.program_id(0)
    kv_head = pid // n_rep

    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + pid * head_dim + dim_offsets).to(tl.float32)

    scale = 1.0 / (head_dim**0.5)

    m = tl.load(m_ptr + pid)
    l = tl.load(l_ptr + pid)

    kv_head_base = kv_head * total_tokens * head_dim

    for start in range(0, total_tokens, block_kv):
        offs = start + tl.arange(0, block_kv)
        token_mask = offs < total_tokens

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
def decode_attention_accumulate_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    m_ptr,
    l_ptr,
    acc_ptr,
    block_mass_ptr,
    block_id_offset,
    total_tokens,
    n_rep: tl.constexpr,
    head_dim: tl.constexpr,
    tokens_per_block: tl.constexpr,
    max_blocks: tl.constexpr,
):
    """
    Pass 2: (m, l) are already final (from decode_attention_stats_kernel, or
    a prior stats pass over every group) and read-only here, so no more
    rescaling is needed -- just accumulate this group's weighted V into acc,
    and separately record each pager-block's own share of the total
    attention mass (needed for the Rust pager's content-aware scoring),
    normalized by the same final l.
    """
    pid = tl.program_id(0)
    kv_head = pid // n_rep

    dim_offsets = tl.arange(0, head_dim)
    q = tl.load(q_ptr + pid * head_dim + dim_offsets).to(tl.float32)

    scale = 1.0 / (head_dim**0.5)

    m_final = tl.load(m_ptr + pid)
    l_final = tl.load(l_ptr + pid)
    acc = tl.load(acc_ptr + pid * head_dim + dim_offsets)

    kv_head_base = kv_head * total_tokens * head_dim
    num_local_blocks = (total_tokens + tokens_per_block - 1) // tokens_per_block

    for local_block in range(0, num_local_blocks):
        start = local_block * tokens_per_block
        offs = start + tl.arange(0, tokens_per_block)
        token_mask = offs < total_tokens

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


def streaming_attention_stats_step(q: torch.Tensor, k: torch.Tensor, m: torch.Tensor, l: torch.Tensor) -> None:
    """Fold one group's K into the running (m, l) stats, in place -- pass 1, no V needed."""
    num_query_heads, head_dim = q.shape
    num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads

    decode_attention_stats_kernel[(num_query_heads,)](
        q, k.contiguous(), m, l, total_tokens, n_rep=n_rep, head_dim=head_dim, block_kv=BLOCK_KV
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
) -> None:
    """Fold one group's K/V into acc and per-pager-block attention mass, in place -- pass 2, (m, l) already final."""
    num_query_heads, head_dim = q.shape
    num_kv_heads, total_tokens, _ = k.shape
    n_rep = num_query_heads // num_kv_heads
    max_blocks = block_mass.shape[1]

    decode_attention_accumulate_kernel[(num_query_heads,)](
        q,
        k.contiguous(),
        v.contiguous(),
        m,
        l,
        acc,
        block_mass,
        block_id_offset,
        total_tokens,
        n_rep=n_rep,
        head_dim=head_dim,
        tokens_per_block=tokens_per_block,
        max_blocks=max_blocks,
    )


def dense_attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, n_rep: int) -> torch.Tensor:
    """Standard scaled_dot_product_attention with GQA head expansion, for comparison."""
    key = k.repeat_interleave(n_rep, dim=0).unsqueeze(0)  # [1, heads, total, head_dim]
    value = v.repeat_interleave(n_rep, dim=0).unsqueeze(0)
    query = q.unsqueeze(0).unsqueeze(2)  # [1, heads, 1, head_dim]

    out = F.scaled_dot_product_attention(query, key, value, is_causal=False)
    return out.squeeze(0).squeeze(1)


def run_case(*, name: str, total_tokens: int, dtype: torch.dtype, rtol: float, atol: float) -> bool:
    torch.manual_seed(0)
    device = torch.device("cuda")
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS

    q = torch.randn(NUM_QUERY_HEADS, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=dtype, device=device)

    triton_out = triton_decode_attention(q, k, v)
    dense_out = dense_attention_reference(q, k, v, n_rep=n_rep).to(dtype)

    max_diff = (triton_out.float() - dense_out.float()).abs().max().item()
    close = torch.allclose(triton_out, dense_out, rtol=rtol, atol=atol)

    print(f"{name:35s} total_tokens={total_tokens:5d} dtype={str(dtype):15s} max_diff={max_diff:.6f} match={close}")
    return close


def run_grouped_case(
    *, name: str, total_tokens: int, group_size: int, dtype: torch.dtype, rtol: float, atol: float
) -> bool:
    """Process the same K/V in several small groups, across several kernel launches, and compare to one big call."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS

    q = torch.randn(NUM_QUERY_HEADS, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=dtype, device=device)

    one_shot_out = triton_decode_attention(q, k, v)

    m, l, acc = streaming_attention_state_init(NUM_QUERY_HEADS, HEAD_DIM, device)
    for start in range(0, total_tokens, group_size):
        end = min(start + group_size, total_tokens)
        streaming_attention_step(q, k[:, start:end, :], v[:, start:end, :], m, l, acc)
    grouped_out = streaming_attention_finalize(acc, l, dtype)

    dense_out = dense_attention_reference(q, k, v, n_rep=n_rep).to(dtype)

    max_diff_vs_one_shot = (grouped_out.float() - one_shot_out.float()).abs().max().item()
    close_vs_dense = torch.allclose(grouped_out, dense_out, rtol=rtol, atol=atol)

    print(
        f"{name:35s} total_tokens={total_tokens:5d} group_size={group_size:4d} dtype={str(dtype):15s} "
        f"max_diff_vs_one_shot={max_diff_vs_one_shot:.6f} match_dense={close_vs_dense}"
    )
    return close_vs_dense


def dense_attention_mass_reference(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, n_rep: int, tokens_per_block: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense attention output plus per-pager-block attention mass (summed softmax weight), for comparison."""
    key = k.repeat_interleave(n_rep, dim=0).float()
    value = v.repeat_interleave(n_rep, dim=0).float()
    scale = 1.0 / (q.shape[-1] ** 0.5)

    scores = torch.einsum("hd,htd->ht", q.float(), key) * scale
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("ht,htd->hd", weights, value)

    total_tokens = k.shape[1]
    num_blocks = (total_tokens + tokens_per_block - 1) // tokens_per_block
    block_mass = torch.stack(
        [weights[:, b * tokens_per_block : (b + 1) * tokens_per_block].sum(dim=-1) for b in range(num_blocks)], dim=1
    )
    return out, block_mass


def run_mass_case(
    *,
    name: str,
    total_tokens: int,
    group_size: int,
    tokens_per_block: int,
    dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> bool:
    """Two-pass (stats, then accumulate-with-mass) attention, processed in small groups, vs. a dense reference."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    n_rep = NUM_QUERY_HEADS // NUM_KV_HEADS

    q = torch.randn(NUM_QUERY_HEADS, HEAD_DIM, dtype=dtype, device=device)
    k = torch.randn(NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=dtype, device=device)
    v = torch.randn(NUM_KV_HEADS, total_tokens, HEAD_DIM, dtype=dtype, device=device)

    num_blocks = (total_tokens + tokens_per_block - 1) // tokens_per_block

    m, l, _ = streaming_attention_state_init(NUM_QUERY_HEADS, HEAD_DIM, device)
    for start in range(0, total_tokens, group_size):
        streaming_attention_stats_step(q, k[:, start : start + group_size, :], m, l)

    acc = torch.zeros(NUM_QUERY_HEADS, HEAD_DIM, dtype=torch.float32, device=device)
    block_mass = torch.zeros(NUM_QUERY_HEADS, num_blocks, dtype=torch.float32, device=device)
    for group_start in range(0, total_tokens, group_size):
        group_end = min(group_start + group_size, total_tokens)
        streaming_attention_accumulate_step(
            q,
            k[:, group_start:group_end, :],
            v[:, group_start:group_end, :],
            m,
            l,
            acc,
            block_mass,
            group_start // tokens_per_block,
            tokens_per_block,
        )

    triton_out = streaming_attention_finalize(acc, l, dtype)
    dense_out, dense_mass = dense_attention_mass_reference(q, k, v, n_rep=n_rep, tokens_per_block=tokens_per_block)

    out_close = torch.allclose(triton_out, dense_out.to(dtype), rtol=rtol, atol=atol)
    mass_diff = (block_mass - dense_mass).abs().max().item()
    mass_close = torch.allclose(block_mass, dense_mass, rtol=rtol, atol=atol)

    print(
        f"{name:35s} total_tokens={total_tokens:5d} group_size={group_size:4d} blocks={num_blocks:4d} "
        f"out_match={out_close} mass_max_diff={mass_diff:.6f} mass_match={mass_close}"
    )
    return out_close and mass_close


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    print("device:", torch.cuda.get_device_name(), torch.cuda.get_device_capability())
    print("num_query_heads:", NUM_QUERY_HEADS, "num_kv_heads:", NUM_KV_HEADS, "head_dim:", HEAD_DIM)
    print()

    results = []

    results.append(run_case(name="single_token", total_tokens=1, dtype=torch.float16, rtol=2e-2, atol=2e-2))
    results.append(run_case(name="ragged_tail_below_block", total_tokens=17, dtype=torch.float16, rtol=2e-2, atol=2e-2))
    results.append(run_case(name="exact_one_block", total_tokens=BLOCK_KV, dtype=torch.float16, rtol=2e-2, atol=2e-2))
    results.append(
        run_case(name="ragged_tail_above_block", total_tokens=BLOCK_KV + 5, dtype=torch.float16, rtol=2e-2, atol=2e-2)
    )
    results.append(run_case(name="few_blocks", total_tokens=200, dtype=torch.float16, rtol=2e-2, atol=2e-2))
    results.append(run_case(name="many_blocks", total_tokens=2000, dtype=torch.float16, rtol=2e-2, atol=2e-2))
    results.append(run_case(name="many_blocks_long", total_tokens=6000, dtype=torch.float16, rtol=2e-2, atol=2e-2))
    results.append(run_case(name="fp32", total_tokens=200, dtype=torch.float32, rtol=1e-4, atol=1e-4))

    results.append(
        run_grouped_case(
            name="grouped_few_groups", total_tokens=200, group_size=64, dtype=torch.float16, rtol=2e-2, atol=2e-2
        )
    )
    results.append(
        run_grouped_case(
            name="grouped_many_small_groups", total_tokens=200, group_size=16, dtype=torch.float16, rtol=2e-2, atol=2e-2
        )
    )
    results.append(
        run_grouped_case(
            name="grouped_long_context", total_tokens=6000, group_size=128, dtype=torch.float16, rtol=2e-2, atol=2e-2
        )
    )
    results.append(
        run_grouped_case(
            name="grouped_ragged_groups", total_tokens=203, group_size=17, dtype=torch.float16, rtol=2e-2, atol=2e-2
        )
    )

    results.append(
        run_mass_case(
            name="mass_few_groups",
            total_tokens=200,
            group_size=64,
            tokens_per_block=16,
            dtype=torch.float16,
            rtol=2e-2,
            atol=2e-2,
        )
    )
    results.append(
        run_mass_case(
            name="mass_many_small_groups",
            total_tokens=200,
            group_size=16,
            tokens_per_block=16,
            dtype=torch.float16,
            rtol=2e-2,
            atol=2e-2,
        )
    )
    results.append(
        run_mass_case(
            name="mass_long_context",
            total_tokens=6000,
            group_size=128,
            tokens_per_block=16,
            dtype=torch.float16,
            rtol=2e-2,
            atol=2e-2,
        )
    )
    results.append(
        run_mass_case(
            name="mass_ragged_tail",
            total_tokens=203,
            group_size=32,
            tokens_per_block=16,
            dtype=torch.float16,
            rtol=2e-2,
            atol=2e-2,
        )
    )
    results.append(
        run_mass_case(
            name="mass_fp32",
            total_tokens=200,
            group_size=64,
            tokens_per_block=16,
            dtype=torch.float32,
            rtol=1e-4,
            atol=1e-4,
        )
    )

    print()
    if all(results):
        print(f"OK: Triton decode-attention kernel matches dense attention in all {len(results)} cases.")
    else:
        print(f"FAIL: {results.count(False)}/{len(results)} cases did not match.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
