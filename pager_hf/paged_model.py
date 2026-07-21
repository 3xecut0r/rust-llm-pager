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
# shape is otherwise identical: Qwen2Attention still uses the older
# rotary_emb(x, seq_len=N) -> full table, apply_rotary_pos_emb(..., position_ids)
# indexes it. Llama and Mistral have both already moved to
# rotary_emb(x, position_ids) -> pre-indexed cos/sin and an apply_rotary_pos_emb
# with no position_ids arg -- Llama's decoder layer precomputes it once and
# shares it across layers via a position_embeddings kwarg; Mistral's decoder
# layer has no such kwarg at all and calls rotary_emb itself inside self_attn,
# which is exactly the fallback branch below already covers. Verified end to
# end (same_token_ids) on Qwen2.5-0.5B-Instruct, TinyLlama-1.1B, and a real
# (if tiny) Mistral checkpoint; other architectures aren't recognized, so they
# raise instead of silently computing something wrong.
_SUPPORTED_STREAMING_MODEL_TYPES = frozenset({"qwen2", "llama", "mistral"})
_LLAMA_STYLE_ROPE_MODEL_TYPES = frozenset({"llama", "mistral"})


def _compute_rope(
    self_attn, model_type: str, query_states, key_states, value_states, position_ids, position_embeddings
):
    """Apply RoPE the way this model family's transformers implementation expects; see the note above."""
    if model_type == "qwen2":
        # cos/sin sized to the true absolute position, not the tail cache's own
        # (much shorter) length -- rotary_emb slices its table to exactly seq_len.
        kv_seq_len = int(position_ids.max().item()) + 1
        cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        return _qwen2_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if model_type in _LLAMA_STYLE_ROPE_MODEL_TYPES:
        # LlamaModel.forward computes (cos, sin) once from *our* position_ids and shares it
        # across every layer, passed in as position_embeddings -- reuse it instead of a second,
        # redundant rotary_emb call. Mistral's decoder layer never provides position_embeddings
        # at all, so it always falls through to calling rotary_emb here directly -- matching
        # what MistralAttention.forward itself does.
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
        # Requires a Qwen2-, Llama-, or Mistral-family decoder-layer structure
        # (q/k/v/o_proj, rotary_emb -- see _SUPPORTED_STREAMING_MODEL_TYPES);
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

            outputs = self.model(
                input_ids=input_id_tensor,
                attention_mask=current_attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                output_attentions=self._needs_attention,
            )

            self._num_blocks, self._tail_past, added_block_ids = self._append_new_full_blocks(
                store=self._store,
                past_key_values=get_legacy_past_key_values(outputs),
                known_num_blocks=self._num_blocks,
            )

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

            query_states = self.q_proj(hidden_states).view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = (
                self.k_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            )
            value_states = (
                self.v_proj(hidden_states).view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            )

            query_states, key_states = _compute_rope(
                self, model_type, query_states, key_states, value_states, position_ids, position_embeddings
            )

            # DynamicCache.update() ignores cache_kwargs entirely (sin/cos/cache_position are only
            # consumed by other Cache subclasses, e.g. StaticCache) -- kept only for API shape.
            cache_kwargs = {"cache_position": cache_position}
            past_key_value.update(key_states, value_states, layer_idx, cache_kwargs)

            q = query_states[:, :, 0, :]  # [batch, num_heads, head_dim]
            m, l, acc = streaming_attention_state_init(bsz, q.shape[1], q.shape[2], device)

            block_ids = list(range(paged_model._num_blocks))
            group_size = paged_model.streaming_group_size_blocks

            # Padding mask, aligned by absolute token position: block i covers
            # [i*tokens_per_block, (i+1)*tokens_per_block), the tail covers
            # everything after the last full block up to and including the new
            # token. Sliced per chunk below so it always matches that chunk's length.
            full_tokens = paged_model._num_blocks * paged_model.tokens_per_block
            padding_mask = paged_model._streaming_attention_mask

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
                    streaming_attention_stats_step(q, group_key, m, l, valid=group_valid)
                    transient_bytes += moved_bytes
                    transient_copies += moved_copies
                    del group_key, group_value
                streaming_attention_stats_step(q, full_tail_key, m, l, valid=tail_valid)

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
                    )
                    transient_bytes += moved_bytes
                    transient_copies += moved_copies
                    del group_key, group_value
                streaming_attention_fold_fixed_step(q, full_tail_key, full_tail_value, m, l, acc, valid=tail_valid)

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
                    streaming_attention_step(q, group_key, group_value, m, l, acc, valid=group_valid)
                    transient_bytes += moved_bytes
                    transient_copies += moved_copies
                    del group_key, group_value
                streaming_attention_step(q, full_tail_key, full_tail_value, m, l, acc, valid=tail_valid)

            paged_model._transient_cpu_to_gpu_bytes += transient_bytes
            paged_model._transient_cpu_to_gpu_copies += transient_copies

            streaming_out = streaming_attention_finalize(acc, l, q.dtype)
            attn_output = streaming_out.unsqueeze(2)  # [batch, num_heads, 1, head_dim]
            attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, self.hidden_size)
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

        past = None
        outputs = None

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            outputs = self.model(
                input_ids=prefix_input_ids[:, start:end],
                attention_mask=prefix_attention_mask[:, :end],
                position_ids=position_ids[:, start:end],
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
