from __future__ import annotations

import logging
import types

import torch
from transformers.cache_utils import DynamicCache

from .kv_block_store import tensor_nbytes
from .kv_utils import get_legacy_past_key_values
from .paged_model import _SUPPORTED_STREAMING_MODEL_TYPES, PagedModel, _compute_rope, _project_qkv

# Sliding-window architectures (Mistral, Gemma2 -- see paged_model._sliding_window_for_layer)
# are explicitly NOT supported here: the single-session path enforces the window via a shared
# scalar "current position" per layer, but batched sessions can each be at a *different* absolute
# position, so the bound would need to be per-row, not a single scalar -- a real additional
# wrinkle not attempted in this pass. Rejected explicitly (see the model_type check below) rather
# than silently computing an unwindowed (wrong) result for these two.
_SLIDING_WINDOW_MODEL_TYPES = frozenset({"mistral", "gemma2"})
from .streaming_attention import (
    streaming_attention_accumulate_step,
    streaming_attention_finalize,
    streaming_attention_fold_fixed_step,
    streaming_attention_state_init,
    streaming_attention_stats_step,
    streaming_attention_step,
)

logger = logging.getLogger(__name__)


def batched_decode_step(
    sessions: list[PagedModel], next_input_ids: list[int], device: torch.device
) -> list[torch.Tensor]:
    """
    Advance N independent, single-row PagedModel sessions by exactly one
    token each, in ONE combined model.forward() call instead of N separate
    ones -- real cross-session continuous batching (Stage 2), not Stage 1's
    round-robin (pager_hf.serving.ContinuousBatchingScheduler). Each session
    keeps its own KVBlockStore/tail_past/Rust pager; only the *compute* (one
    shared forward pass) is batched, not the memory-placement decisions.

    Scope of this first pass, each enforced explicitly below rather than
    silently assumed: every session must share the same underlying model,
    tokens_per_block, and streaming_group_size_blocks; use
    use_streaming_attention=True; be single-row (batch_size==1, no internal
    padding); already have an active generate() session; use the same kind
    of policy (attention-scoring or not, so every row goes through the same
    one-pass-vs-two-pass kernel path); and -- to keep the per-session block
    alignment a clean whole-block offset instead of a fractional one --
    currently have the *same tail length*. Sessions naturally converge to a
    shared set of tail lengths over time (every active session's tail grows
    by exactly one token per step and rolls into a new block at the same
    modulus), so a caller groups sessions by tail length per round; sessions
    at a different history length in whole blocks are still batched
    together fine via padding, which is the actual point of this mechanism.

    Returns each session's last-position logits, in the same order as
    `sessions`, so the caller decides greedy/sampling exactly like
    generate()'s own loop does with _forward_step's output.
    """
    if not sessions:
        return []

    model = sessions[0].model
    tokens_per_block = sessions[0].tokens_per_block
    group_size = sessions[0].streaming_group_size_blocks
    num_layers = sessions[0]._num_layers
    tail_len = sessions[0]._tail_past[0][0].shape[2]
    needs_attention_flag = sessions[0]._needs_attention

    for session in sessions:
        if session.model is not model:
            raise ValueError("batched_decode_step requires every session to share the same underlying model.")
        if not session.use_streaming_attention:
            raise ValueError("batched_decode_step only supports use_streaming_attention=True sessions.")
        if session._store is None:
            raise ValueError("batched_decode_step requires every session to already have an active generate() session.")
        if session.tokens_per_block != tokens_per_block:
            raise ValueError("batched_decode_step requires every session to share the same tokens_per_block.")
        if session.streaming_group_size_blocks != group_size:
            raise ValueError(
                "batched_decode_step requires every session to share the same streaming_group_size_blocks."
            )
        if session._tail_past[0][0].shape[0] != 1:
            raise ValueError("batched_decode_step only supports single-row (batch_size==1) sessions.")
        if session._tail_past[0][0].shape[2] != tail_len:
            raise ValueError(
                "batched_decode_step requires every session to currently have the same tail length "
                f"(got {session._tail_past[0][0].shape[2]}, expected {tail_len}); "
                "group sessions by tail length per round."
            )
        if session._needs_attention != needs_attention_flag:
            raise ValueError(
                "batched_decode_step requires every session to use the same kind of policy "
                "(attention-scoring or not) -- they drive different kernel passes."
            )

    model_type = getattr(model.config, "model_type", None)
    if model_type not in _SUPPORTED_STREAMING_MODEL_TYPES:
        raise NotImplementedError(f"batched_decode_step doesn't recognize model_type={model_type!r}.")
    if model_type in _SLIDING_WINDOW_MODEL_TYPES:
        raise NotImplementedError(
            f"batched_decode_step doesn't yet support model_type={model_type!r}'s sliding-window attention -- "
            "batched sessions can be at different absolute positions, so the window bound would need to be "
            "per-row, not a single scalar (see paged_model._sliding_window_for_layer). Use PagedModel.generate() "
            "directly instead (Stage 1 round-robin serving, which calls generate() per session, is unaffected)."
        )

    num_sessions = len(sessions)
    max_blocks = max(session._num_blocks for session in sessions)
    pad_blocks = [max_blocks - session._num_blocks for session in sessions]

    total_len = max_blocks * tokens_per_block + tail_len + 1
    attention_mask = torch.zeros((num_sessions, total_len), dtype=torch.long, device=device)
    for row, session in enumerate(sessions):
        real_len = session._num_blocks * tokens_per_block + tail_len + 1
        attention_mask[row, total_len - real_len :] = 1

    input_ids = torch.tensor(next_input_ids, dtype=torch.long, device=device).unsqueeze(1)
    position_ids = PagedModel._position_ids_from_mask(attention_mask)[:, -1:]
    cache_position = torch.full((1,), int(position_ids.max().item()), dtype=torch.long, device=device)

    # Combined per-layer tail: every session's own single-row tail, concatenated
    # along the batch dim -- valid because every session was checked to share
    # the same tail_len above, so no padding is needed here (unlike the block
    # history, whose lengths genuinely differ and are padded via pad_blocks).
    combined_tail = [
        (
            torch.cat([session._tail_past[layer][0] for session in sessions], dim=0),
            torch.cat([session._tail_past[layer][1] for session in sessions], dim=0),
        )
        for layer in range(num_layers)
    ]
    cache = DynamicCache.from_legacy_cache(tuple(combined_tail))

    if needs_attention_flag:
        for session in sessions:
            session._mass_accum = torch.zeros(session._num_blocks, dtype=torch.float32, device=device)

    transient_state = {"bytes": 0, "copies": 0}
    block_positions = list(range(max_blocks))

    def build_layer_forward(layer_idx: int):
        def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            cache_position=None,
            position_embeddings=None,
            **kwargs,
        ):
            bsz, q_len, _ = hidden_states.size()
            if q_len != 1:
                raise RuntimeError("batched streaming forward expects one token at a time (a decode step).")

            query_states, key_states, value_states = _project_qkv(self, model_type, hidden_states, bsz, q_len)

            query_states, key_states = _compute_rope(
                self, model_type, query_states, key_states, value_states, position_ids, position_embeddings
            )

            # DynamicCache.update() ignores cache_kwargs entirely -- kept only for API shape,
            # same as the single-session forward this generalizes.
            cache_kwargs = {"cache_position": cache_position}
            past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

            q = query_states[:, :, 0, :]  # [num_sessions, num_heads, head_dim]
            m, l, acc = streaming_attention_state_init(bsz, q.shape[1], q.shape[2], device)

            tail_key, tail_value = combined_tail[layer_idx]
            full_tail_key = torch.cat([tail_key, key_states], dim=2)
            full_tail_value = torch.cat([tail_value, value_states], dim=2)
            tail_valid = torch.ones((num_sessions, full_tail_key.shape[2]), dtype=torch.int32, device=device)

            kv_heads = self.num_key_value_heads
            head_dim = self.head_dim

            def gather_group(local_positions: list[int]):
                """
                One group's K/V across every session: local_positions are
                measured from the front of the combined (max_blocks-wide)
                numbering. A session whose own history doesn't reach a given
                position yet (own_block_id < 0, i.e. still padding) gets a
                zero block with an all-invalid row instead -- the same
                mechanism that already excludes a shorter row's padding
                within one session's own batch, reused across sessions here.
                """
                key_rows, value_rows, valid_rows = [], [], []
                for row, session in enumerate(sessions):
                    session_keys, session_values, session_valid = [], [], []
                    for pos in local_positions:
                        own_block_id = pos - pad_blocks[row]
                        if 0 <= own_block_id < session._num_blocks:
                            block_key, block_value = session._store.get_any(own_block_id)
                            was_on_gpu = session._store.has_gpu(own_block_id)
                            layer_key = block_key[layer_idx].to(device, non_blocking=False)
                            layer_value = block_value[layer_idx].to(device, non_blocking=False)
                            if not was_on_gpu:
                                transient_state["bytes"] += tensor_nbytes(layer_key) + tensor_nbytes(layer_value)
                                transient_state["copies"] += 1
                            session_keys.append(layer_key.permute(0, 2, 1, 3))
                            session_values.append(layer_value.permute(0, 2, 1, 3))
                            session_valid.append(torch.ones(1, tokens_per_block, dtype=torch.int32, device=device))
                        else:
                            session_keys.append(
                                torch.zeros(1, kv_heads, tokens_per_block, head_dim, dtype=q.dtype, device=device)
                            )
                            session_values.append(
                                torch.zeros(1, kv_heads, tokens_per_block, head_dim, dtype=q.dtype, device=device)
                            )
                            session_valid.append(torch.zeros(1, tokens_per_block, dtype=torch.int32, device=device))
                    key_rows.append(torch.cat(session_keys, dim=2))
                    value_rows.append(torch.cat(session_values, dim=2))
                    valid_rows.append(torch.cat(session_valid, dim=1))
                return torch.cat(key_rows, dim=0), torch.cat(value_rows, dim=0), torch.cat(valid_rows, dim=0)

            def group_row_active(i: int, group_len: int) -> torch.Tensor:
                """
                [num_sessions] 0/1: does this session have ANY real (non-padding)
                block in local positions [i, i+group_len)? Padding is always a
                prefix (pad_blocks[row] wide) in the combined max_blocks-wide
                numbering, so a row has real data here iff its padding prefix
                ends before the group does. A row with active=0 lets the kernel
                skip this group's KV loop entirely instead of processing a chunk
                that gather_group already filled with all-invalid padding.
                """
                return torch.tensor(
                    [1 if pad_blocks[row] < i + group_len else 0 for row in range(num_sessions)],
                    dtype=torch.int32,
                    device=device,
                )

            if needs_attention_flag:
                for i in range(0, len(block_positions), group_size):
                    positions = block_positions[i : i + group_size]
                    group_key, group_value, group_valid = gather_group(positions)
                    row_active = group_row_active(i, len(positions))
                    streaming_attention_stats_step(q, group_key, m, l, valid=group_valid, active=row_active)
                    del group_key, group_value
                streaming_attention_stats_step(q, full_tail_key, m, l, valid=tail_valid)

                block_mass = torch.zeros(bsz, q.shape[1], max(max_blocks, 1), dtype=torch.float32, device=device)
                for i in range(0, len(block_positions), group_size):
                    positions = block_positions[i : i + group_size]
                    group_key, group_value, group_valid = gather_group(positions)
                    row_active = group_row_active(i, len(positions))
                    streaming_attention_accumulate_step(
                        q,
                        group_key,
                        group_value,
                        m,
                        l,
                        acc,
                        block_mass,
                        i,
                        tokens_per_block,
                        valid=group_valid,
                        active=row_active,
                    )
                    del group_key, group_value
                streaming_attention_fold_fixed_step(q, full_tail_key, full_tail_value, m, l, acc, valid=tail_valid)

                # Average over heads only, keep the per-session row -- unlike the
                # single-session forward (which also averages over batch, since
                # every row there belongs to the SAME session), each row here is a
                # DIFFERENT, independent session and must accumulate its own mass
                # separately, never mixed with another session's.
                per_row_mass = block_mass.mean(dim=1)  # [num_sessions, max_blocks]
                for row, session in enumerate(sessions):
                    if session._num_blocks > 0:
                        offset = pad_blocks[row]
                        session._mass_accum += per_row_mass[row, offset : offset + session._num_blocks]
            else:
                for i in range(0, len(block_positions), group_size):
                    positions = block_positions[i : i + group_size]
                    group_key, group_value, group_valid = gather_group(positions)
                    row_active = group_row_active(i, len(positions))
                    streaming_attention_step(q, group_key, group_value, m, l, acc, valid=group_valid, active=row_active)
                    del group_key, group_value
                streaming_attention_step(q, full_tail_key, full_tail_value, m, l, acc, valid=tail_valid)

            streaming_out = streaming_attention_finalize(acc, l, q.dtype)
            attn_output = streaming_out.unsqueeze(2)
            # -1, not self.hidden_size -- see the matching comment in paged_model.py's
            # _build_streaming_layer_forward (Gemma2's head_dim isn't hidden_size//num_heads).
            # Currently unreachable for gemma2 specifically (rejected above for its sliding
            # window), but keeps this generalization consistent rather than a latent landmine.
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)
            return attn_output, None, past_key_value

        return forward

    originals = []
    for layer_idx, layer in enumerate(model.model.layers):
        originals.append(layer.self_attn.forward)
        layer.self_attn.forward = types.MethodType(build_layer_forward(layer_idx), layer.self_attn)

    try:
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )
    finally:
        for layer, original_forward in zip(model.model.layers, originals):
            layer.self_attn.forward = original_forward

    grown_tail = get_legacy_past_key_values(outputs)  # per layer: [num_sessions, kv_heads, tail_len+1, head_dim]

    for row, session in enumerate(sessions):
        session_grown_tail = [(key[row : row + 1], value[row : row + 1]) for key, value in grown_tail]
        session._num_blocks, session._tail_past, added_block_ids = session._append_new_full_blocks_from_grown_tail(
            store=session._store, grown_tail_past_key_values=session_grown_tail, known_num_blocks=session._num_blocks
        )

        if session._num_blocks == 0:
            continue

        if needs_attention_flag:
            block_attention = session._finalize_streaming_block_attention()
        else:
            block_attention = [1.0 / session._num_blocks] * session._num_blocks

        query_block = session._num_blocks - 1
        session._rust_pager.on_step(query_block, 0, block_attention)
        if added_block_ids:
            session._rust_pager.force_rebalance(query_block)
        session._store.apply_tiers(session._rust_pager.tiers(), device)

    logger.debug(
        "batched_decode_step: num_sessions=%d max_blocks=%d transient_bytes=%d transient_copies=%d",
        num_sessions,
        max_blocks,
        transient_state["bytes"],
        transient_state["copies"],
    )

    return [outputs.logits[row, -1, :] for row in range(num_sessions)]
