from __future__ import annotations

import logging
import threading
import types
from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as _llama_apply_rotary_pos_emb
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb as _qwen2_apply_rotary_pos_emb

import pager as _pager_rs

from .kv_block_store import KVBlockStore, tensor_nbytes
from .kv_utils import (
    extract_last_query_block_attention,
    extract_single_block_from_past,
    get_legacy_past_key_values,
    real_past_to_blocks,
    reconstruct_past_from_store,
    split_full_blocks_and_tail,
)
from .streaming_attention import (
    streaming_attention_accumulate_step,
    streaming_attention_finalize,
    streaming_attention_fold_fixed_step,
    streaming_attention_state_init,
    streaming_attention_stats_step,
    streaming_attention_step,
)

logger = logging.getLogger(__name__)

# Policies whose block placement does not use attention scores at all
# (see pager/src/core.rs: place_baseline_blocks only sorts by pinned + id).
# Any other policy name (including unrecognized ones, matching the Rust
# side's own fallback-to-heavy_hitter behavior) is treated as needing a
# real attention signal.
_POLICIES_WITHOUT_ATTENTION_SIGNAL = frozenset({"recent_only", "sinks_recent"})

# use_streaming_attention patches self_attn.forward, so it needs to know the
# decoder layer's attribute layout (q/k/v/o_proj, rotary_emb -- shared across
# every family below) *and* which RoPE calling convention this transformers
# version uses for that family. In 4.44.2 they differ even though the layer
# shape is otherwise identical: Qwen2Attention (and Qwen2MoE/Starcoder2, whose
# self_attn is a straight copy of it) uses the older
# rotary_emb(x, seq_len=N) -> full table, apply_rotary_pos_emb(..., position_ids)
# indexes it. Llama, Mistral, Gemma, Phi3, and Gemma2 have all already moved to
# rotary_emb(x, position_ids) -> pre-indexed cos/sin and an apply_rotary_pos_emb
# with no position_ids arg -- Llama's decoder layer precomputes it once and
# shares it across layers via a position_embeddings kwarg; Mistral's, Gemma's,
# Phi3's, and Gemma2's decoder layers have no such kwarg at all and call
# rotary_emb themselves inside self_attn, which is exactly the fallback branch
# below already covers (confirmed Phi3RotaryEmbedding/apply_rotary_pos_emb are
# byte-identical to Llama's own, despite Phi3Attention.forward passing an extra
# unused seq_len= to rotary_emb -- Phi3RotaryEmbedding.forward never reads it).
# Verified end to end (same_token_ids) on Qwen2.5-0.5B-Instruct, TinyLlama-1.1B,
# and real (if tiny) Mistral/Qwen2MoE/Starcoder2/Gemma/Phi3/Gemma2 checkpoints;
# other architectures aren't recognized, so they raise instead of silently
# computing something wrong.
#
# Phi3 and Gemma2 needed real handling beyond the RoPE dispatch, done below:
# Phi3 fuses q/k/v into one qkv_proj linear (_project_qkv splits it); Gemma2
# uses a non-default QK^T scale (query_pre_attn_scalar, not 1/sqrt(head_dim))
# and attn-logit softcapping baked into the raw scores (_scale_and_softcap),
# plus real sliding-window attention on alternating layers (_sliding_window_for_layer)
# -- which also turned out to be a real, silent gap in the already-shipped Mistral
# support (its eager-mode HF baseline enforces a uniform sliding_window=4096 via
# the model-level causal mask; the streaming path here never truncated for it,
# since it builds its own padding-only mask and ignores the incoming
# attention_mask entirely). Fixed for both, single-session path only -- see
# _sliding_window_for_layer's docstring for why batched_decode.py's
# cross-session path isn't covered here.
#
# Still deliberately NOT included: Phi (not Phi3; o_proj is named "dense"),
# StableLm (partial rotary -- only a head_dim slice gets rotated), Olmo/Cohere
# (optional config-gated qk clipping / per-head qk-norm that would silently
# compute a wrong result if unhandled) -- each needs its own explicit handling,
# not just a frozenset entry.
_SUPPORTED_STREAMING_MODEL_TYPES = frozenset(
    {"qwen2", "qwen2_moe", "starcoder2", "llama", "mistral", "gemma", "phi3", "gemma2"}
)
_LLAMA_STYLE_ROPE_MODEL_TYPES = frozenset({"llama", "mistral", "gemma", "phi3", "gemma2"})
_QWEN2_STYLE_ROPE_MODEL_TYPES = frozenset({"qwen2", "qwen2_moe", "starcoder2"})


def _compute_rope(
    self_attn, model_type: str, query_states, key_states, value_states, position_ids, position_embeddings
):
    """Apply RoPE the way this model family's transformers implementation expects; see the note above."""
    if model_type in _QWEN2_STYLE_ROPE_MODEL_TYPES:
        # cos/sin sized to the true absolute position, not the tail cache's own
        # (much shorter) length -- rotary_emb slices its table to exactly seq_len.
        kv_seq_len = int(position_ids.max().item()) + 1
        cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        return _qwen2_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if model_type in _LLAMA_STYLE_ROPE_MODEL_TYPES:
        # LlamaModel.forward computes (cos, sin) once from *our* position_ids and shares it
        # across every layer, passed in as position_embeddings -- reuse it instead of a second,
        # redundant rotary_emb call. Mistral's and Gemma's decoder layers never provide
        # position_embeddings at all, so they always fall through to calling rotary_emb
        # here directly -- matching what MistralAttention/GemmaAttention.forward do themselves.
        if position_embeddings is not None:
            cos, sin = position_embeddings
        else:
            cos, sin = self_attn.rotary_emb(value_states, position_ids)
        return _llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)

    raise NotImplementedError(
        f"use_streaming_attention doesn't recognize model_type={model_type!r}. Only "
        f"{sorted(_SUPPORTED_STREAMING_MODEL_TYPES)} have been verified against a baseline so far "
        "-- pass use_streaming_attention=False for other architectures."
    )


def _project_qkv(self_attn, model_type: str, hidden_states: torch.Tensor, bsz: int, q_len: int):
    """
    Compute (query_states, key_states, value_states), viewed and transposed to
    [batch, heads, q_len, head_dim]. Every supported architecture except Phi3
    keeps q_proj/k_proj/v_proj as separate linears; Phi3 fuses them into one
    qkv_proj linear (Phi3Attention.forward splits it the same way below).
    """
    if model_type == "phi3":
        qkv = self_attn.qkv_proj(hidden_states)
        query_pos = self_attn.num_heads * self_attn.head_dim
        kv_pos = self_attn.num_key_value_heads * self_attn.head_dim
        q = qkv[..., :query_pos]
        k = qkv[..., query_pos : query_pos + kv_pos]
        v = qkv[..., query_pos + kv_pos :]
    else:
        q = self_attn.q_proj(hidden_states)
        k = self_attn.k_proj(hidden_states)
        v = self_attn.v_proj(hidden_states)

    query_states = q.view(bsz, q_len, self_attn.num_heads, self_attn.head_dim).transpose(1, 2)
    key_states = k.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    value_states = v.view(bsz, q_len, self_attn.num_key_value_heads, self_attn.head_dim).transpose(1, 2)
    return query_states, key_states, value_states


def _scale_and_softcap(self_attn, model_type: str) -> tuple[float | None, float | None]:
    """Gemma2 uses a non-default QK^T scale (query_pre_attn_scalar-based, already computed by
    Gemma2Attention.__init__ as self.scaling -- not recomputed here) and attn-logit softcapping
    (may genuinely be None, meaning disabled). Every other architecture uses the streaming
    kernels' own defaults (1/sqrt(head_dim), no softcap) -- returning (None, None) for those."""
    if model_type == "gemma2":
        return self_attn.scaling, self_attn.config.attn_logit_softcapping
    return None, None


def _sliding_window_for_layer(self_attn, model_type: str) -> int | None:
    """
    None means full/global attention for this layer -- the common case. Only Mistral and Gemma2
    need a real window bound in this transformers version:

    - Gemma2 runs sliding-window attention on alternating layers; Gemma2Attention.__init__ already
      computed the per-layer value as self.sliding_window (None for the global layers) -- read it
      directly rather than re-deriving the even/odd rule here.
    - Mistral applies its config.sliding_window uniformly to every layer. Its own eager-mode HF
      baseline (MistralModel._update_causal_mask) really does enforce this via the causal mask --
      confirmed by reading that function's source -- so the streaming path here was silently
      diverging from the true baseline whenever context exceeded it (default 4096), the whole time
      Mistral has been "supported". This fixes that, not just Gemma2's version of the same gap.
    - Qwen2/Qwen2MoE/Starcoder2 are NOT included: their sliding-window config defaults to
      off, and Qwen2Model._update_causal_mask (the eager-mode mask builder actually used here,
      confirmed by reading it) doesn't reference sliding-window at all in this transformers
      version -- the branch that does exists only in the flash-attention-2-specific forward, never
      exercised by this project. There's no real baseline divergence to fix for them.
    """
    if model_type == "gemma2":
        return self_attn.sliding_window
    if model_type == "mistral":
        return self_attn.config.sliding_window
    return None


@dataclass
class GenerationStats:
    total_new_blocks: int = 0
    final_num_blocks: int = 0
    total_gpu_to_cpu_mb: float = 0.0
    total_cpu_to_gpu_mb: float = 0.0
    total_gpu_to_cpu_copies: int = 0
    total_cpu_to_gpu_copies: int = 0
    mean_attention_in_gpu: float = 0.0
    min_attention_in_gpu: float = 0.0
    max_attention_in_gpu: float = 0.0


class PagedModel:
    # Wraps a HuggingFace causal LM so its KV cache is paged between GPU
    # and CPU memory by the Rust pager, instead of staying fully resident
    # in VRAM. Greedy by default, sampling available via do_sample=True. A
    # PagedModel instance holds one mutable session (see "Persistence across
    # generate() calls" in the README), so generate()/reset() reject
    # concurrent calls from another thread instead of racing on that state.

    def __init__(
        self,
        model,
        *,
        vram_budget: int,
        ram_budget: int,
        recent_window: int = 64,
        rebalance_interval: int = 4,
        promote_margin: float = 0.05,
        ram_promote_margin: float = 0.20,
        policy: str = "sinks_heavy_hitter",
        tokens_per_block: int = 16,
        prefill_chunk_tokens: int = 512,
        use_streaming_attention: bool = True,
        streaming_group_size_blocks: int = 64,
    ):
        self.model = model
        self.tokens_per_block = tokens_per_block
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self._needs_attention = policy not in _POLICIES_WITHOUT_ATTENTION_SIGNAL
        self._ram_budget = ram_budget

        # Default on: decode steps never reconstruct the full KV cache into one
        # GPU-resident buffer (the "reload everything, then attend" path
        # below). Instead every layer streams its attention in
        # streaming_group_size_blocks-sized groups straight out of the
        # KVBlockStore, wherever each group currently lives. On this project's
        # target hardware/model, group size barely affects peak GPU memory
        # (the gather buffer is tiny next to the model's own footprint) but
        # larger groups mean fewer kernel launches and are meaningfully
        # faster -- see bench/streaming_group_size_sweep_mvp.py; 64 is past
        # the point of diminishing returns there while still bounding the
        # gather buffer for bigger contexts/models this hasn't been tested on.
        # Requires a Qwen2/Qwen2MoE/Starcoder2/Llama/Mistral/Gemma-family
        # decoder-layer structure (q/k/v/o_proj, rotary_emb -- see _SUPPORTED_STREAMING_MODEL_TYPES);
        # other architectures raise a clear NotImplementedError from generate()
        # instead of silently computing something wrong -- pass
        # use_streaming_attention=False for those.
        # batch_size > 1 is supported for every policy, streaming or not --
        # for heavy_hitter/sinks_heavy_hitter, block placement is one decision
        # shared by the whole batch, scored by an average across rows (see
        # extract_last_query_block_attention / the mass reduction below).
        self.use_streaming_attention = use_streaming_attention
        self.streaming_group_size_blocks = streaming_group_size_blocks
        self._mass_accum: torch.Tensor | None = None
        self._transient_cpu_to_gpu_bytes = 0
        self._transient_cpu_to_gpu_copies = 0
        self._streaming_attention_mask: torch.Tensor | None = None

        self._pager_kwargs = dict(
            vram=vram_budget,
            ram=ram_budget,
            recent=recent_window,
            rebalance_interval=rebalance_interval,
            promote_margin=promote_margin,
            ram_promote_margin=ram_promote_margin,
            policy=policy,
        )

        self.last_run_stats: GenerationStats | None = None

        self._store: KVBlockStore | None = None
        self._rust_pager = None
        self._num_layers: int | None = None
        self._num_blocks: int | None = None
        self._tail_past = None
        self._primed_tokens = 0
        self._primed_prefix: torch.Tensor | None = None
        self._lock = threading.Lock()

    def _acquire_or_raise(self) -> None:
        """Fail fast instead of silently racing when another call is already in flight."""
        if not self._lock.acquire(blocking=False):
            logger.warning(
                "generate()/reset() rejected: another call is already in flight on this PagedModel instance."
            )
            raise RuntimeError(
                "Another generate() or reset() call is already running on this "
                "PagedModel instance. It holds one mutable session, so concurrent "
                "calls from another thread would corrupt its state rather than "
                "run as independent requests. Use a separate PagedModel per "
                "concurrent session, or synchronize calls to this one yourself."
            )

    def reset(self) -> None:
        """Drop the current session; the next generate() call starts fresh."""
        self._acquire_or_raise()
        try:
            if self._store is not None:
                logger.info("PagedModel session reset (was at %d blocks).", self._num_blocks)
            self._store = None
            self._rust_pager = None
            self._num_layers = None
            self._num_blocks = None
            self._tail_past = None
            self._primed_tokens = 0
            self._primed_prefix = None
        finally:
            self._lock.release()

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        *,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ) -> list[int] | list[list[int]]:
        """
        Generate up to max_new_tokens tokens, continuing any previous session on this model.

        Greedy by default. Pass do_sample=True for temperature/top_k/top_p
        sampling; generator, if given, must be on the same device as the model.
        """
        self._acquire_or_raise()
        try:
            device = input_ids.device
            batch_size = input_ids.shape[0]
            self._validate_left_padded(attention_mask)

            if do_sample:
                if temperature <= 0:
                    raise ValueError(f"temperature must be > 0 for sampling, got {temperature}.")
                if top_k is not None and top_k < 1:
                    raise ValueError(f"top_k must be >= 1, got {top_k}.")
                if top_p is not None and not (0.0 < top_p <= 1.0):
                    raise ValueError(f"top_p must be in (0, 1], got {top_p}.")

            session_freshly_started = self._store is None
            if session_freshly_started:
                self._start_session(input_ids, attention_mask)

            streaming_patch_originals = self._patch_layers_for_streaming() if self.use_streaming_attention else None
            try:
                if not session_freshly_started:
                    self._extend_session(input_ids, attention_mask, device)

                next_input_id = input_ids[:, -1:]
                current_attention_mask = attention_mask

                generated_per_row: list[list[int]] = [[] for _ in range(batch_size)]
                stats = GenerationStats()
                attention_in_gpu_values: list[float] = []

                for _ in range(max_new_tokens):
                    outputs, step_stats = self._forward_step(next_input_id, current_attention_mask, device)

                    if do_sample:
                        next_token_id = self._sample_next_token(
                            outputs.logits[:, -1, :],
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            generator=generator,
                        )
                    else:
                        next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)  # [batch_size, 1]
                    for row_tokens, token_id in zip(generated_per_row, next_token_id[:, 0].tolist()):
                        row_tokens.append(token_id)

                    stats.total_new_blocks += step_stats["new_blocks"]
                    stats.total_gpu_to_cpu_mb += step_stats["gpu_to_cpu_mb"]
                    stats.total_cpu_to_gpu_mb += step_stats["cpu_to_gpu_mb"]
                    stats.total_gpu_to_cpu_copies += step_stats["gpu_to_cpu_copies"]
                    stats.total_cpu_to_gpu_copies += step_stats["cpu_to_gpu_copies"]
                    attention_in_gpu_values.append(step_stats["attention_in_gpu"])

                    next_input_id = next_token_id
                    current_attention_mask = torch.cat(
                        [
                            current_attention_mask,
                            torch.ones(
                                (current_attention_mask.shape[0], 1),
                                dtype=current_attention_mask.dtype,
                                device=current_attention_mask.device,
                            ),
                        ],
                        dim=1,
                    )
            finally:
                if streaming_patch_originals is not None:
                    self._unpatch_layers(streaming_patch_originals)

            # The very last token of (input_ids + generated) is only ever
            # sampled, never forward-passed, so it's excluded from "primed".
            full_sequence = torch.cat(
                [input_ids, torch.tensor(generated_per_row, dtype=input_ids.dtype, device=input_ids.device)], dim=1
            )
            self._primed_prefix = full_sequence[:, :-1].detach().clone()
            self._primed_tokens = full_sequence.shape[-1] - 1

            stats.final_num_blocks = self._num_blocks
            if attention_in_gpu_values:
                stats.mean_attention_in_gpu = sum(attention_in_gpu_values) / len(attention_in_gpu_values)
                stats.min_attention_in_gpu = min(attention_in_gpu_values)
                stats.max_attention_in_gpu = max(attention_in_gpu_values)

            self.last_run_stats = stats

            return generated_per_row[0] if batch_size == 1 else generated_per_row
        finally:
            self._lock.release()

    @torch.inference_mode()
    def generate_beam_search(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_beams: int,
        max_new_tokens: int,
        *,
        eos_token_id: int | None = None,
        length_penalty: float = 1.0,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        generator: torch.Generator | None = None,
    ):
        """
        Beam search: keeps num_beams parallel hypotheses per input prompt,
        each with its own KV-cache history in the pager, and at every step
        picks the best num_beams continuations across all beams of the same
        prompt combined -- not the best continuation per row independently,
        which is what generate() does for a real batch. That selection can
        send one beam's history into more than one new row next step (a
        strong beam spawning multiple children) or into none at all (a weak
        beam dying), so this drives its own decode loop instead of
        generate()'s, reusing _forward_step (the single-token forward plus
        block/tier bookkeeping) unchanged for each row-advance.

        input_ids can be a batch of independent prompts (batch_size > 1):
        each runs its own num_beams-wide search independently -- top-k (or,
        under do_sample, sampling) selection happens separately per prompt,
        candidates are never mixed across different prompts.

        length_penalty divides a candidate sequence's cumulative
        log-probability by its length**length_penalty at final-selection
        time only (not during the per-step search itself); default 1.0
        matches HuggingFace's own default (>1 favors longer sequences, <1
        favors shorter, 0 is pure cumulative log-probability with no length
        adjustment at all). Only length_penalty=0.0 is verified byte-exact
        against HuggingFace's own beam search reference (bench/real_kv_beam_search_mvp.py)
        -- HuggingFace's default nonzero case additionally re-normalizes
        scores throughout its own separate per-step hypothesis bookkeeping,
        which isn't replicated here; nonzero length_penalty here changes the
        final choice in the expected direction but isn't guaranteed
        bit-for-bit identical to HuggingFace's.

        num_return_sequences (1..num_beams) returns more than one final
        candidate per prompt, ranked by length-penalized score, most likely
        first.

        do_sample=True draws each step's num_beams continuations from the
        (temperature/top_k/top_p-filtered) joint candidate distribution via
        sampling without replacement, instead of the deterministic top-k --
        a real but much rarer "stochastic beam search" mode; greedy
        (do_sample=False) is the default and the only mode verified against
        HuggingFace's reference.

        A beam that already emitted eos_token_id is "frozen" -- forced to
        keep emitting only eos_token_id at zero further score change --
        rather than dropped, so it stays selectable without shrinking beam
        count mid-search.

        Doesn't support resuming this session via a later generate() call
        (unlike generate()'s own persistence contract) -- every call starts
        fresh.

        Returns (matching generate()'s batch_size convention):
        - batch_size == 1, num_return_sequences == 1: list[int]
        - batch_size == 1, num_return_sequences > 1: list[list[int]]
        - batch_size > 1, num_return_sequences == 1: list[list[int]]
        - batch_size > 1, num_return_sequences > 1: list[list[list[int]]]
        """
        self._acquire_or_raise()
        try:
            batch_size = input_ids.shape[0]
            if num_beams < 1:
                raise ValueError(f"num_beams must be >= 1, got {num_beams}")
            if not 1 <= num_return_sequences <= num_beams:
                raise ValueError(
                    f"num_return_sequences must be between 1 and num_beams ({num_beams}), got {num_return_sequences}"
                )
            if do_sample:
                if temperature <= 0:
                    raise ValueError(f"temperature must be > 0 for sampling, got {temperature}.")
                if top_k is not None and top_k < 1:
                    raise ValueError(f"top_k must be >= 1, got {top_k}.")
                if top_p is not None and not (0.0 < top_p <= 1.0):
                    raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
            self._validate_left_padded(attention_mask)

            device = input_ids.device
            total_rows = batch_size * num_beams
            beam_input_ids = input_ids.repeat_interleave(num_beams, dim=0)
            beam_attention_mask = attention_mask.repeat_interleave(num_beams, dim=0)

            self._start_session(beam_input_ids, beam_attention_mask)
            streaming_patch_originals = self._patch_layers_for_streaming() if self.use_streaming_attention else None

            try:
                next_input_id = beam_input_ids[:, -1:]
                current_attention_mask = beam_attention_mask

                # Every beam within a prompt's group starts as an identical copy of
                # that prompt, so their first-step logits are identical too -- if
                # every beam_score started at 0, the top-k/sampling below would treat
                # several duplicate rows as if they were distinct, independent
                # hypotheses. Only the first beam of each group starts "active"
                # (score 0); every other beam starts at -inf, so -inf + anything
                # stays -inf and the first step's candidates for each prompt can only
                # come from that one real distribution -- standard beam-search init.
                beam_scores = torch.full((total_rows,), float("-inf"), dtype=torch.float32, device=device)
                beam_scores[0::num_beams] = 0.0
                beam_tokens: list[list[int]] = [[] for _ in range(total_rows)]
                beam_finished = [False] * total_rows

                for _ in range(max_new_tokens):
                    outputs, _ = self._forward_step(next_input_id, current_attention_mask, device)
                    log_probs = torch.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)  # [total_rows, vocab]
                    vocab_size = log_probs.shape[-1]

                    # A finished beam may only "continue" with eos_token_id, at zero
                    # additional score -- keeps it selectable without letting it keep
                    # growing its score by emitting further real tokens.
                    if eos_token_id is not None:
                        for row in range(total_rows):
                            if beam_finished[row]:
                                frozen = torch.full_like(log_probs[row], float("-inf"))
                                frozen[eos_token_id] = 0.0
                                log_probs[row] = frozen

                    if do_sample:
                        scaled = log_probs / temperature
                        if top_k is not None:
                            kth_value = torch.topk(scaled, min(top_k, vocab_size), dim=-1).values[:, -1, None]
                            scaled = scaled.masked_fill(scaled < kth_value, float("-inf"))
                        if top_p is not None:
                            sorted_scaled, sorted_indices = torch.sort(scaled, descending=True, dim=-1)
                            sorted_probs = torch.softmax(sorted_scaled, dim=-1)
                            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                            sorted_scaled = sorted_scaled.masked_fill(
                                cumulative_probs - sorted_probs > top_p, float("-inf")
                            )
                            scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_indices, sorted_scaled)
                        candidate_log_probs = torch.log_softmax(scaled, dim=-1)
                    else:
                        candidate_log_probs = log_probs

                    # [total_rows, vocab] -> [batch_size, num_beams * vocab]: top-k (or
                    # sampling) below operates per prompt group, never mixing
                    # candidates from one prompt's beams into another's selection.
                    candidate_scores = (beam_scores.unsqueeze(1) + candidate_log_probs).view(
                        batch_size, num_beams * vocab_size
                    )

                    if do_sample:
                        probs = torch.softmax(candidate_scores, dim=-1)
                        local_indices = torch.multinomial(probs, num_beams, replacement=False, generator=generator)
                        top_scores = torch.gather(candidate_scores, 1, local_indices)
                    else:
                        top_scores, local_indices = torch.topk(candidate_scores, num_beams, dim=-1)

                    local_parent_beams = local_indices // vocab_size  # [batch_size, num_beams]
                    chosen_tokens = local_indices % vocab_size  # [batch_size, num_beams]
                    group_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * num_beams
                    global_parent_beams = (local_parent_beams + group_offsets).flatten().tolist()
                    chosen_tokens_flat = chosen_tokens.flatten().tolist()

                    beam_tokens = [
                        beam_tokens[parent] + [token] for parent, token in zip(global_parent_beams, chosen_tokens_flat)
                    ]
                    beam_finished = [
                        beam_finished[parent] or (eos_token_id is not None and token == eos_token_id)
                        for parent, token in zip(global_parent_beams, chosen_tokens_flat)
                    ]
                    beam_scores = top_scores.flatten()

                    self._reorder_batch(global_parent_beams)

                    next_input_id = torch.tensor(chosen_tokens_flat, dtype=input_ids.dtype, device=device).unsqueeze(1)
                    current_attention_mask = torch.cat(
                        [
                            current_attention_mask[global_parent_beams],
                            torch.ones((total_rows, 1), dtype=current_attention_mask.dtype, device=device),
                        ],
                        dim=1,
                    )

                    if all(beam_finished):
                        break
            finally:
                if streaming_patch_originals is not None:
                    self._unpatch_layers(streaming_patch_originals)

            # Final selection: rank each prompt's num_beams candidates by
            # length-penalized score, independently per prompt group.
            per_prompt_results: list[list[list[int]]] = []
            for group in range(batch_size):
                scored = []
                for row in range(group * num_beams, (group + 1) * num_beams):
                    tokens = beam_tokens[row]
                    if eos_token_id is not None and eos_token_id in tokens:
                        tokens = tokens[: tokens.index(eos_token_id) + 1]
                    normalized_score = float(beam_scores[row]) / (max(len(tokens), 1) ** length_penalty)
                    scored.append((normalized_score, tokens))
                scored.sort(key=lambda item: item[0], reverse=True)
                per_prompt_results.append([tokens for _, tokens in scored[:num_return_sequences]])

            logger.info(
                "generate_beam_search finished: batch_size=%d num_beams=%d num_return_sequences=%d",
                batch_size,
                num_beams,
                num_return_sequences,
            )

            if batch_size == 1 and num_return_sequences == 1:
                return per_prompt_results[0][0]
            if batch_size == 1:
                return per_prompt_results[0]
            if num_return_sequences == 1:
                return [group[0] for group in per_prompt_results]
            return per_prompt_results
        finally:
            self._lock.release()

    def _reorder_batch(self, new_row_indices: list[int]) -> None:
        """Reindex every batch row of live session state (block store + tail) to new_row_indices."""
        self._store.reorder_batch_rows(new_row_indices)

        index = torch.tensor(new_row_indices, dtype=torch.long)
        self._tail_past = [
            (key.index_select(0, index.to(key.device)), value.index_select(0, index.to(value.device)))
            for key, value in self._tail_past
        ]

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        *,
        temperature: float,
        top_k: int | None,
        top_p: float | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        """Sample one token per row from logits, after temperature/top-k/top-p filtering."""
        logits = logits / temperature

        if top_k is not None:
            kth_value = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth_value, float("-inf"))

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # A token is outside the nucleus once the probability mass strictly
            # before it already reaches top_p, so it isn't needed to hit top_p.
            sorted_logits = sorted_logits.masked_fill(cumulative_probs - sorted_probs > top_p, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    def _start_session(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
        """Prime the KV store and pager from scratch, using all but the last token as prefix."""
        outputs = self._chunked_prefill(input_ids[:, :-1], attention_mask[:, :-1])

        self._store, self._num_layers, self._num_blocks, self._tail_past = self._build_initial_store(
            get_legacy_past_key_values(outputs)
        )
        self._rust_pager = _pager_rs.PyPager(**self._pager_kwargs)

        logger.info(
            "Started PagedModel session: batch_size=%d prompt_tokens=%d layers=%d blocks=%d policy=%r " "streaming=%s",
            input_ids.shape[0],
            input_ids.shape[-1],
            self._num_layers,
            self._num_blocks,
            self._pager_kwargs["policy"],
            self.use_streaming_attention,
        )

    def _extend_session(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device) -> None:
        """Feed any tokens not yet primed through the model before generate() decodes new ones."""
        total_len = input_ids.shape[-1]

        if total_len - 1 < self._primed_tokens:
            raise ValueError(
                "input_ids is shorter than the already-primed session "
                f"({total_len - 1} < {self._primed_tokens} tokens). "
                "Call reset() to start a new session instead of rewinding one."
            )

        if not torch.equal(input_ids[:, : self._primed_tokens].cpu(), self._primed_prefix.cpu()):
            raise ValueError(
                "input_ids diverges from the already-primed session prefix. "
                "generate() expects each call to extend the same growing "
                "sequence (previous input + previously generated tokens + "
                "any new tokens). Call reset() to start a different session."
            )

        pending = input_ids[:, self._primed_tokens : total_len - 1]

        for i in range(pending.shape[-1]):
            self._forward_step(pending[:, i : i + 1], attention_mask[:, : self._primed_tokens + i + 1], device)

    def _forward_step(self, input_id_tensor: torch.Tensor, current_attention_mask: torch.Tensor, device: torch.device):
        """Run one token through the model, register any new blocks, and re-place tiers."""
        if self.use_streaming_attention:
            outputs, reload_bytes, reload_copies = self._forward_step_streaming(
                input_id_tensor, current_attention_mask, device
            )
            self._num_blocks, self._tail_past, added_block_ids = self._append_new_full_blocks_from_grown_tail(
                store=self._store,
                grown_tail_past_key_values=get_legacy_past_key_values(outputs),
                known_num_blocks=self._num_blocks,
            )
        else:
            reload_bytes, reload_copies = self._reload_all_blocks(self._store, device, self._num_blocks)

            cache = DynamicCache.from_legacy_cache(
                tuple(
                    reconstruct_past_from_store(
                        store=self._store,
                        num_layers=self._num_layers,
                        num_blocks=self._num_blocks,
                        tail_past=self._tail_past,
                    )
                )
            )
            position_ids = self._position_ids_from_mask(current_attention_mask)[:, -1:]
            # Explicit, not derived by HF's own default (torch.arange(0, q_len), i.e.
            # always 0 for a single-token decode step regardless of how much is
            # already cached) -- the same reasoning _forward_step_streaming's own
            # cache_position already documents. Reload mode never hit this before
            # because every previously-supported architecture's causal-mask builder
            # happens not to depend on cache_position's absolute value for a plain
            # DynamicCache; Gemma2's does (_update_causal_mask uses it directly),
            # so an unset cache_position (defaulting to 0) built the wrong mask and
            # silently produced a wrong-but-plausible result -- found via real-model
            # verification, not by inspection.
            cache_position = torch.full((1,), int(position_ids.max().item()), dtype=torch.long, device=device)

            outputs = self.model(
                input_ids=input_id_tensor,
                attention_mask=current_attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                past_key_values=cache,
                use_cache=True,
                output_attentions=self._needs_attention,
            )

            self._num_blocks, self._tail_past, added_block_ids = self._append_new_full_blocks(
                store=self._store,
                past_key_values=get_legacy_past_key_values(outputs),
                known_num_blocks=self._num_blocks,
            )

        # A prompt shorter than one tokens_per_block leaves _num_blocks at 0
        # for the first several decode steps (everything still lives in the
        # tail, nothing promoted to a full pager block yet). There's nothing
        # for the Rust pager to score, pin, or rebalance in that state --
        # query_block = self._num_blocks - 1 would go negative, and every
        # policy's block_attention computation below assumes at least one
        # block exists (a uniform 1/num_blocks vector divides by zero;
        # attention extraction indexes a block that isn't there yet).
        if self._num_blocks == 0:
            block_attention: list[float] = []
            summary_before = self._store.summary()
            summary_after = summary_before
        else:
            if self._needs_attention and self.use_streaming_attention:
                block_attention = self._finalize_streaming_block_attention()
            elif self._needs_attention:
                block_attention = extract_last_query_block_attention(
                    outputs, tokens_per_block=self.tokens_per_block, num_blocks=self._num_blocks
                )
            else:
                # placement doesn't use scores for this policy; a uniform
                # vector satisfies the pager API without needing attentions.
                block_attention = [1.0 / self._num_blocks] * self._num_blocks

            query_block = self._num_blocks - 1

            summary_before = self._store.summary()
            self._rust_pager.on_step(query_block, 0, block_attention)
            if added_block_ids:
                self._rust_pager.force_rebalance(query_block)
            self._store.apply_tiers(self._rust_pager.tiers(), device)
            summary_after = self._store.summary()

        attention_in_gpu = sum(
            block_attention[block_id] for block_id in self._store.gpu_block_ids() if block_id < len(block_attention)
        )

        step_stats = {
            "new_blocks": len(added_block_ids),
            "gpu_to_cpu_mb": (summary_after["gpu_to_cpu_bytes"] - summary_before["gpu_to_cpu_bytes"]) / 1_000_000,
            "cpu_to_gpu_mb": reload_bytes / 1_000_000,
            "gpu_to_cpu_copies": summary_after["gpu_to_cpu_copies"] - summary_before["gpu_to_cpu_copies"],
            "cpu_to_gpu_copies": reload_copies,
            "attention_in_gpu": attention_in_gpu,
        }

        logger.debug(
            "step: num_blocks=%d new_blocks=%d gpu_to_cpu_mb=%.3f cpu_to_gpu_mb=%.3f attention_in_gpu=%.4f",
            self._num_blocks,
            step_stats["new_blocks"],
            step_stats["gpu_to_cpu_mb"],
            step_stats["cpu_to_gpu_mb"],
            step_stats["attention_in_gpu"],
        )

        return outputs, step_stats

    def _forward_step_streaming(
        self, input_id_tensor: torch.Tensor, current_attention_mask: torch.Tensor, device: torch.device
    ):
        """
        Run one token through the model with every layer's self_attn patched
        to stream attention from the KVBlockStore, instead of reconstructing
        the full context into one GPU-resident cache first. past_key_values
        only ever holds the small tail (not the historical blocks), so
        cache_position must be passed explicitly -- HF would otherwise derive
        it from the tail cache's own (much shorter, irrelevant) length.
        """
        position_ids = self._position_ids_from_mask(current_attention_mask)[:, -1:]
        # A shared upper-bound cache_position, not a per-row one: rows can have
        # different real positions (different padding amounts), but they all
        # grow the same stacked cache tensor in lockstep at the column level.
        cache_position = torch.full((1,), int(position_ids.max().item()), dtype=torch.long, device=device)
        cache = DynamicCache.from_legacy_cache(tuple(self._tail_past))

        if self._needs_attention:
            self._mass_accum = torch.zeros(self._num_blocks, dtype=torch.float32, device=device)
        self._transient_cpu_to_gpu_bytes = 0
        self._transient_cpu_to_gpu_copies = 0
        # Raw 0/1 padding mask, read by the patched layer forward to exclude
        # padded positions from attention -- HF's outer forward would otherwise
        # transform this into a 4D additive mask before it reaches self_attn,
        # which our kernel doesn't consume, so the original is kept on the side.
        self._streaming_attention_mask = current_attention_mask

        outputs = self.model(
            input_ids=input_id_tensor,
            attention_mask=current_attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )

        return outputs, self._transient_cpu_to_gpu_bytes, self._transient_cpu_to_gpu_copies

    def _finalize_streaming_block_attention(self) -> list[float]:
        """Average this step's per-layer block-mass accumulation into a normalized per-block score list."""
        if self._num_blocks == 0 or self._mass_accum is None:
            return [1.0 / self._num_blocks] * self._num_blocks if self._num_blocks else []

        averaged = (self._mass_accum / self._num_layers).tolist()
        total = sum(averaged)

        if total <= 0:
            return [1.0 / self._num_blocks] * self._num_blocks

        return [x / total for x in averaged]

    def _gather_layer_kv_group(
        self, *, layer_idx: int, block_ids: list[int], device: torch.device
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int, int]:
        """
        Gather one layer's K/V for the given block ids into one small
        contiguous [batch, kv_heads, group_tokens, head_dim] buffer, reading
        each block from whichever tier it currently lives on via get_any --
        without moving it or otherwise changing the pager's placement
        decision. Returns (key, value, transient_bytes, transient_copies)
        for whatever had to be temporarily copied from CPU.

        Per-block tensors are collected as permuted views (cheap, no copy)
        and combined with a single torch.cat instead of one
        slice-assignment per block. Profiling on a real 7B/8000-token
        decode step found this loop -- one small copy-kernel launch per
        block, per tensor, per layer, per step -- was the actual
        throughput bottleneck (64% of total decode time), not the
        attention kernel (15%): the same per-tiny-op launch-overhead
        pattern that made the project's very first pure-Python prototype
        impractical, resurfacing here in the gather path instead.
        """
        if not block_ids:
            return None, None, 0, 0

        key_parts = []
        value_parts = []
        transient_bytes = 0
        transient_copies = 0

        for block_id in block_ids:
            block_key, block_value = self._store.get_any(block_id)
            was_on_gpu = self._store.has_gpu(block_id)

            layer_key = block_key[layer_idx].to(device, non_blocking=False)  # [batch, block_len, kv_heads, head_dim]
            layer_value = block_value[layer_idx].to(device, non_blocking=False)

            if not was_on_gpu:
                transient_bytes += tensor_nbytes(layer_key) + tensor_nbytes(layer_value)
                transient_copies += 1

            key_parts.append(layer_key.permute(0, 2, 1, 3))  # [batch, kv_heads, block_len, head_dim]
            value_parts.append(layer_value.permute(0, 2, 1, 3))

        dest_key = torch.cat(key_parts, dim=2)
        dest_value = torch.cat(value_parts, dim=2)

        return dest_key, dest_value, transient_bytes, transient_copies

    def _build_streaming_layer_forward(self, layer_idx: int, device: torch.device, model_type: str):
        """One decode-time forward for one decoder layer: stream historical blocks in groups, fold in the tail."""
        paged_model = self

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
                raise RuntimeError("Streaming attention forward expects one token at a time (a decode step).")

            query_states, key_states, value_states = _project_qkv(self, model_type, hidden_states, bsz, q_len)

            query_states, key_states = _compute_rope(
                self, model_type, query_states, key_states, value_states, position_ids, position_embeddings
            )

            # DynamicCache.update() ignores cache_kwargs entirely (sin/cos/cache_position are only
            # consumed by other Cache subclasses, e.g. StaticCache) -- kept only for API shape.
            cache_kwargs = {"cache_position": cache_position}
            past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

            q = query_states[:, :, 0, :]  # [batch, num_heads, head_dim]
            m, l, acc = streaming_attention_state_init(bsz, q.shape[1], q.shape[2], device)
            scale, softcap = _scale_and_softcap(self, model_type)

            block_ids = list(range(paged_model._num_blocks))
            group_size = paged_model.streaming_group_size_blocks

            # Padding mask, aligned by absolute token position: block i covers
            # [i*tokens_per_block, (i+1)*tokens_per_block), the tail covers
            # everything after the last full block up to and including the new
            # token. Sliced per chunk below so it always matches that chunk's length.
            full_tokens = paged_model._num_blocks * paged_model.tokens_per_block
            padding_mask = paged_model._streaming_attention_mask

            # Sliding-window architectures (Mistral, Gemma2 -- see
            # _sliding_window_for_layer): exclude any position further back than
            # the window from *every* chunk below, by folding the exclusion into
            # padding_mask itself once, here -- every group_valid/tail_valid slice
            # downstream already derives from padding_mask, so this single AND
            # covers both the block-group loop and the tail fold for free.
            sliding_window = _sliding_window_for_layer(self, model_type)
            if sliding_window is not None:
                current_pos = int(position_ids.max().item())
                col_positions = torch.arange(padding_mask.shape[1], device=device)
                window_valid = (col_positions > (current_pos - sliding_window)).unsqueeze(0)
                padding_mask = padding_mask.to(torch.int32) * window_valid.to(torch.int32)

            tail_key, tail_value = paged_model._tail_past[layer_idx]
            full_tail_key = torch.cat([tail_key, key_states], dim=2)
            full_tail_value = torch.cat([tail_value, value_states], dim=2)
            tail_valid = padding_mask[:, full_tokens:].to(torch.int32)

            transient_bytes = 0
            transient_copies = 0

            if paged_model._needs_attention:
                for i in range(0, len(block_ids), group_size):
                    group_len = len(block_ids[i : i + group_size])
                    group_key, group_value, moved_bytes, moved_copies = paged_model._gather_layer_kv_group(
                        layer_idx=layer_idx, block_ids=block_ids[i : i + group_size], device=device
                    )
                    group_start_tok = i * paged_model.tokens_per_block
                    group_valid = padding_mask[
                        :, group_start_tok : group_start_tok + group_len * paged_model.tokens_per_block
                    ].to(torch.int32)
                    streaming_attention_stats_step(q, group_key, m, l, valid=group_valid, scale=scale, softcap=softcap)
                    transient_bytes += moved_bytes
                    transient_copies += moved_copies
                    del group_key, group_value
                streaming_attention_stats_step(q, full_tail_key, m, l, valid=tail_valid, scale=scale, softcap=softcap)

                block_mass = torch.zeros(
                    bsz, q.shape[1], max(paged_model._num_blocks, 1), dtype=torch.float32, device=device
                )
                for i in range(0, len(block_ids), group_size):
                    group_len = len(block_ids[i : i + group_size])
                    group_key, group_value, moved_bytes, moved_copies = paged_model._gather_layer_kv_group(
                        layer_idx=layer_idx, block_ids=block_ids[i : i + group_size], device=device
                    )
                    group_start_tok = i * paged_model.tokens_per_block
                    group_valid = padding_mask[
                        :, group_start_tok : group_start_tok + group_len * paged_model.tokens_per_block
                    ].to(torch.int32)
                    streaming_attention_accumulate_step(
                        q,
                        group_key,
                        group_value,
                        m,
                        l,
                        acc,
                        block_mass,
                        i,
                        paged_model.tokens_per_block,
                        valid=group_valid,
                        scale=scale,
                        softcap=softcap,
                    )
                    transient_bytes += moved_bytes
                    transient_copies += moved_copies
                    del group_key, group_value
                streaming_attention_fold_fixed_step(
                    q, full_tail_key, full_tail_value, m, l, acc, valid=tail_valid, scale=scale, softcap=softcap
                )

                if paged_model._num_blocks > 0:
                    # Average over batch rows too, not just heads: block placement is one decision
                    # shared by the whole batch, so at batch_size > 1 this is a compromise score
                    # across rows rather than any single row's own preference -- harmless for
                    # correctness (every row's attention always covers the full context regardless
                    # of where blocks physically sit), it only affects how well-suited the shared
                    # placement is to each row.
                    paged_model._mass_accum += block_mass.mean(dim=(0, 1))[: paged_model._num_blocks]
            else:
                for i in range(0, len(block_ids), group_size):
                    group_len = len(block_ids[i : i + group_size])
                    group_key, group_value, moved_bytes, moved_copies = paged_model._gather_layer_kv_group(
                        layer_idx=layer_idx, block_ids=block_ids[i : i + group_size], device=device
                    )
                    group_start_tok = i * paged_model.tokens_per_block
                    group_valid = padding_mask[
                        :, group_start_tok : group_start_tok + group_len * paged_model.tokens_per_block
                    ].to(torch.int32)
                    streaming_attention_step(
                        q, group_key, group_value, m, l, acc, valid=group_valid, scale=scale, softcap=softcap
                    )
                    transient_bytes += moved_bytes
                    transient_copies += moved_copies
                    del group_key, group_value
                streaming_attention_step(
                    q, full_tail_key, full_tail_value, m, l, acc, valid=tail_valid, scale=scale, softcap=softcap
                )

            paged_model._transient_cpu_to_gpu_bytes += transient_bytes
            paged_model._transient_cpu_to_gpu_copies += transient_copies

            streaming_out = streaming_attention_finalize(acc, l, q.dtype)
            attn_output = streaming_out.unsqueeze(2)  # [batch, num_heads, 1, head_dim]
            # -1, not self.hidden_size: Gemma2 configures head_dim independently of
            # hidden_size // num_heads (e.g. hidden_size=2304, num_heads*head_dim=2048
            # on a real Gemma2-2B), so num_heads*head_dim can genuinely differ from
            # hidden_size -- confirmed Gemma2Attention.forward itself uses
            # .view(bsz, q_len, -1) here, not self.hidden_size. A no-op reshape target
            # for every other architecture, where the two are always equal by
            # construction.
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        return forward

    def _patch_layers_for_streaming(self) -> list:
        """Patch every decoder layer's self_attn.forward to stream from the KVBlockStore; return the originals."""
        model_type = getattr(self.model.config, "model_type", None)
        if model_type not in _SUPPORTED_STREAMING_MODEL_TYPES:
            logger.error("use_streaming_attention rejected: unrecognized model_type=%r.", model_type)
            raise NotImplementedError(
                f"use_streaming_attention doesn't recognize model_type={model_type!r}. Only "
                f"{sorted(_SUPPORTED_STREAMING_MODEL_TYPES)} have been verified against a baseline so far "
                "-- pass use_streaming_attention=False for other architectures."
            )

        device = next(self.model.parameters()).device
        originals = []

        for layer_idx, layer in enumerate(self.model.model.layers):
            originals.append(layer.self_attn.forward)
            layer.self_attn.forward = types.MethodType(
                self._build_streaming_layer_forward(layer_idx, device, model_type), layer.self_attn
            )

        logger.info(
            "Streaming attention active: model_type=%r, %d layers patched, group_size=%d blocks.",
            model_type,
            len(originals),
            self.streaming_group_size_blocks,
        )

        return originals

    def _unpatch_layers(self, originals: list) -> None:
        for layer, original_forward in zip(self.model.model.layers, originals):
            layer.self_attn.forward = original_forward

    def _chunked_prefill(self, prefix_input_ids: torch.Tensor, prefix_attention_mask: torch.Tensor):
        """Prefill a long prefix in pieces so HuggingFace never computes logits for the whole thing at once."""
        seq_len = prefix_input_ids.shape[-1]
        chunk_size = self.prefill_chunk_tokens
        position_ids = self._position_ids_from_mask(prefix_attention_mask)

        # An explicit empty DynamicCache, not None: every other supported model's
        # own XModel.forward auto-wraps a None past_key_values into a fresh
        # DynamicCache when use_cache=True, so passing one explicitly is a no-op
        # for them -- but Gemma2Model.forward (confirmed by reading its source)
        # has no such auto-wrapping at all; it just passes past_key_values
        # straight through and returns it unchanged as next_cache. With
        # past=None that means Gemma2Attention.forward's `if past_key_value is
        # not None: past_key_value.update(...)` never runs, nothing ever gets
        # cached, and the model returns past_key_values=None -- not a
        # streaming-attention bug, a real gap in the plain prefill path that
        # would have hit Gemma2 even with use_streaming_attention=False.
        past = DynamicCache()
        outputs = None

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            # Explicit, absolute cache_position -- HF's own default
            # (torch.arange(0, chunk_len)) restarts from 0 for every chunk,
            # which is wrong for any chunk after the first (same class of bug
            # as _forward_step's reload branch; see that fix's comment for why
            # this only visibly matters for Gemma2 so far, on prompts longer
            # than prefill_chunk_tokens).
            outputs = self.model(
                input_ids=prefix_input_ids[:, start:end],
                attention_mask=prefix_attention_mask[:, :end],
                position_ids=position_ids[:, start:end],
                cache_position=torch.arange(start, end, device=prefix_input_ids.device),
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
            )
            past = outputs.past_key_values

        return outputs

    @staticmethod
    def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        """Compute RoPE position ids from attention_mask, skipping padded columns."""
        position_ids = attention_mask.long().cumsum(-1) - 1
        return position_ids.masked_fill(attention_mask == 0, 0)

    @staticmethod
    def _validate_left_padded(attention_mask: torch.Tensor) -> None:
        """Raise if attention_mask isn't left-padding (zeros then ones) per row."""
        mask = attention_mask.bool()

        if not mask.any(dim=1).all():
            raise NotImplementedError("attention_mask has a row with no real (non-padding) tokens.")

        if not torch.all(mask[:, :-1] <= mask[:, 1:]):
            raise NotImplementedError(
                "PagedModel only supports left-padding (zero or more leading "
                "zeros, then all ones, per row); right-padding or "
                "interleaved masking is not supported."
            )

    def _build_initial_store(self, past_key_values):
        """Build a fresh KV block store from the model's past_key_values."""
        full_past, tail_past, _ = split_full_blocks_and_tail(past_key_values, tokens_per_block=self.tokens_per_block)
        kv_blocks = real_past_to_blocks(full_past, tokens_per_block=self.tokens_per_block)

        store = KVBlockStore(tokens_per_block=self.tokens_per_block, ram_budget_bytes=self._ram_budget)
        for block_id, (key, value) in enumerate(kv_blocks):
            store.put_gpu(block_id, key, value)

        return store, len(full_past), len(kv_blocks), tail_past

    def _append_new_full_blocks(self, *, store: KVBlockStore, past_key_values, known_num_blocks: int):
        """Register any newly completed KV blocks in the store."""
        full_past, tail_past, full_tokens = split_full_blocks_and_tail(
            past_key_values, tokens_per_block=self.tokens_per_block
        )
        new_num_blocks = full_tokens // self.tokens_per_block

        added_block_ids = list(range(known_num_blocks, new_num_blocks))
        for block_id in added_block_ids:
            store.put_gpu(
                block_id,
                *extract_single_block_from_past(full_past, block_id=block_id, tokens_per_block=self.tokens_per_block),
            )

        return new_num_blocks, tail_past, added_block_ids

    def _append_new_full_blocks_from_grown_tail(
        self, *, store: KVBlockStore, grown_tail_past_key_values, known_num_blocks: int
    ):
        """
        Register any newly completed blocks from a grown *tail-only* cache
        (streaming mode never holds the historical blocks in past_key_values,
        so split_full_blocks_and_tail here operates on the tail's own small,
        self-contained length -- new full blocks are extracted at local
        indices within it, then registered at the true global block id).
        """
        newly_full_past, remaining_tail, newly_full_tokens = split_full_blocks_and_tail(
            grown_tail_past_key_values, tokens_per_block=self.tokens_per_block
        )

        added_block_ids = []
        num_blocks = known_num_blocks

        if newly_full_tokens > 0:
            for key, value in real_past_to_blocks(newly_full_past, tokens_per_block=self.tokens_per_block):
                store.put_gpu(num_blocks, key, value)
                added_block_ids.append(num_blocks)
                num_blocks += 1

        return num_blocks, remaining_tail, added_block_ids

    @staticmethod
    def _reload_all_blocks(store: KVBlockStore, device: torch.device, num_blocks: int) -> tuple[int, int]:
        """Make sure every block is back on GPU before the next forward pass."""
        before = store.summary()
        for block_id in range(num_blocks):
            store.ensure_gpu(block_id, device)
        after = store.summary()

        return (
            after["cpu_to_gpu_bytes"] - before["cpu_to_gpu_bytes"],
            after["cpu_to_gpu_copies"] - before["cpu_to_gpu_copies"],
        )
