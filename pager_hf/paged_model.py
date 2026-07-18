from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.cache_utils import DynamicCache

import pager as _pager_rs

from .kv_block_store import KVBlockStore
from .kv_utils import (
    append_tail_to_reconstructed_past,
    extract_last_query_block_attention,
    extract_single_block_from_past,
    get_legacy_past_key_values,
    real_past_to_blocks,
    reconstruct_past_from_store,
    split_full_blocks_and_tail,
)


# Policies whose block placement does not use attention scores at all
# (see pager/src/core.rs: place_baseline_blocks only sorts by pinned + id).
# Any other policy name (including unrecognized ones, matching the Rust
# side's own fallback-to-heavy_hitter behavior) is treated as needing a
# real attention signal.
_POLICIES_WITHOUT_ATTENTION_SIGNAL = frozenset({"recent_only", "sinks_recent"})


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
    """
    Wraps a HuggingFace causal LM so its KV cache is paged between GPU and
    CPU memory by the Rust pager, instead of staying fully resident in VRAM.

    Greedy decoding only for now. The KV block store and pager persist
    across `generate()` calls: each call is given the full sequence so far
    (previous input + previously generated tokens + any new tokens), and
    only the delta beyond what was already processed is fed through the
    model. This is what makes a multi-turn session cheap — the first call
    pays for the full prefill, later calls only pay for what's new.

    Call `reset()` to drop the session and start over from scratch.

    batch_size > 1 is supported for one real forward pass across the whole
    batch (not a Python loop over rows), but only under two constraints:
    every row must be the same length (attention_mask all ones, no padding),
    and the policy must not need per-step attention scores (recent_only,
    sinks_recent). This works because those policies place blocks purely by
    recency/position, so every row in an equal-length batch gets the exact
    same placement decision — one shared Pager and one shared KVBlockStore
    (whose blocks now carry a batch dimension) is enough; heavy_hitter /
    sinks_heavy_hitter would need per-row placement and per-row attention
    extraction, which isn't implemented yet.

    The prefix is prefilled in chunks (`prefill_chunk_tokens`) instead of
    one big forward call, since HuggingFace computes logits for every
    prefilled position at once and a large vocabulary makes that alone
    OOM long before KV-cache residency would matter. For policies that
    don't use attention scores for placement (`recent_only`,
    `sinks_recent`), `output_attentions` is also skipped entirely during
    decoding, since requesting it forces eager attention, which does not
    scale to long context either.
    """

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
    ):
        self.model = model
        self.tokens_per_block = tokens_per_block
        self.prefill_chunk_tokens = prefill_chunk_tokens
        self._needs_attention = policy not in _POLICIES_WITHOUT_ATTENTION_SIGNAL

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

    def reset(self) -> None:
        """Drop the current session; the next generate() call starts fresh."""
        self._store = None
        self._rust_pager = None
        self._num_layers = None
        self._num_blocks = None
        self._tail_past = None
        self._primed_tokens = 0
        self._primed_prefix = None

    @torch.inference_mode()
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            max_new_tokens: int,
    ) -> list[int] | list[list[int]]:
        device = input_ids.device
        batch_size = input_ids.shape[0]

        if batch_size > 1 and self._needs_attention:
            raise NotImplementedError(
                "batch_size > 1 is only supported for policies that don't "
                "score blocks by attention (recent_only, sinks_recent), "
                "since those are the only ones validated at batch>1 so far."
            )
        if not torch.all(attention_mask == 1):
            raise NotImplementedError(
                "PagedModel does not support padding yet; attention_mask "
                "must be all ones (every row the same length)."
            )

        if self._store is None:
            self._start_session(input_ids, attention_mask)
        else:
            self._extend_session(input_ids, attention_mask, device)

        next_input_id = input_ids[:, -1:]
        current_attention_mask = attention_mask

        generated_per_row: list[list[int]] = [[] for _ in range(batch_size)]
        stats = GenerationStats()
        attention_in_gpu_values: list[float] = []

        for _ in range(max_new_tokens):
            outputs, step_stats = self._forward_step(
                next_input_id, current_attention_mask, device
            )

            next_token_id = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True
            )  # [batch_size, 1]
            for row in range(batch_size):
                generated_per_row[row].append(int(next_token_id[row, 0].item()))

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

        # The very last token of (input_ids + generated) is only ever
        # sampled, never forward-passed, so it's excluded from "primed".
        generated_tensor = torch.tensor(
            generated_per_row, dtype=input_ids.dtype, device=input_ids.device
        )
        full_sequence = torch.cat([input_ids, generated_tensor], dim=1)

        self._primed_prefix = full_sequence[:, :-1].detach().clone()
        self._primed_tokens = full_sequence.shape[-1] - 1

        stats.final_num_blocks = self._num_blocks
        if attention_in_gpu_values:
            stats.mean_attention_in_gpu = sum(attention_in_gpu_values) / len(
                attention_in_gpu_values
            )
            stats.min_attention_in_gpu = min(attention_in_gpu_values)
            stats.max_attention_in_gpu = max(attention_in_gpu_values)

        self.last_run_stats = stats

        return generated_per_row[0] if batch_size == 1 else generated_per_row

    def _start_session(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> None:
        prefix_input_ids = input_ids[:, :-1]
        prefix_attention_mask = attention_mask[:, :-1]

        outputs = self._chunked_prefill(prefix_input_ids, prefix_attention_mask)
        current_past = get_legacy_past_key_values(outputs)

        store, num_layers, num_blocks, tail_past = self._build_initial_store(
            current_past
        )

        self._store = store
        self._num_layers = num_layers
        self._num_blocks = num_blocks
        self._tail_past = tail_past
        self._rust_pager = _pager_rs.PyPager(**self._pager_kwargs)

    def _extend_session(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            device: torch.device,
    ) -> None:
        total_len = input_ids.shape[-1]

        if total_len - 1 < self._primed_tokens:
            raise ValueError(
                "input_ids is shorter than the already-primed session "
                f"({total_len - 1} < {self._primed_tokens} tokens). "
                "Call reset() to start a new session instead of rewinding one."
            )

        already_primed = input_ids[:, : self._primed_tokens]
        if not torch.equal(already_primed.cpu(), self._primed_prefix.cpu()):
            raise ValueError(
                "input_ids diverges from the already-primed session prefix. "
                "generate() expects each call to extend the same growing "
                "sequence (previous input + previously generated tokens + "
                "any new tokens). Call reset() to start a different session."
            )

        pending = input_ids[:, self._primed_tokens: total_len - 1]

        for i in range(pending.shape[-1]):
            token = pending[:, i: i + 1]
            mask_so_far = attention_mask[:, : self._primed_tokens + i + 1]
            self._forward_step(token, mask_so_far, device)

    def _forward_step(
            self,
            input_id_tensor: torch.Tensor,
            current_attention_mask: torch.Tensor,
            device: torch.device,
    ):
        reload_bytes, reload_copies = self._reload_all_blocks(
            self._store, device, self._num_blocks
        )

        reconstructed_full_past = reconstruct_past_from_store(
            store=self._store,
            num_layers=self._num_layers,
            num_blocks=self._num_blocks,
        )
        current_past_for_forward = append_tail_to_reconstructed_past(
            reconstructed_full_past,
            self._tail_past,
        )
        cache = DynamicCache.from_legacy_cache(tuple(current_past_for_forward))

        outputs = self.model(
            input_ids=input_id_tensor,
            attention_mask=current_attention_mask,
            past_key_values=cache,
            use_cache=True,
            output_attentions=self._needs_attention,
        )

        current_past = get_legacy_past_key_values(outputs)
        old_num_blocks = self._num_blocks
        self._num_blocks, self._tail_past, added_block_ids = (
            self._append_new_full_blocks(
                store=self._store,
                past_key_values=current_past,
                known_num_blocks=old_num_blocks,
            )
        )

        if self._needs_attention:
            block_attention = extract_last_query_block_attention(
                outputs,
                tokens_per_block=self.tokens_per_block,
                num_blocks=self._num_blocks,
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
            block_attention[block_id]
            for block_id in self._store.gpu_block_ids()
            if block_id < len(block_attention)
        )

        step_stats = {
            "new_blocks": len(added_block_ids),
            "gpu_to_cpu_mb": (
                summary_after["gpu_to_cpu_bytes"] - summary_before["gpu_to_cpu_bytes"]
            ) / 1_000_000,
            "cpu_to_gpu_mb": reload_bytes / 1_000_000,
            "gpu_to_cpu_copies": (
                summary_after["gpu_to_cpu_copies"] - summary_before["gpu_to_cpu_copies"]
            ),
            "cpu_to_gpu_copies": reload_copies,
            "attention_in_gpu": attention_in_gpu,
        }

        return outputs, step_stats

    def _chunked_prefill(
            self,
            prefix_input_ids: torch.Tensor,
            prefix_attention_mask: torch.Tensor,
    ):
        seq_len = prefix_input_ids.shape[-1]
        chunk_size = self.prefill_chunk_tokens

        past = None
        outputs = None

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            outputs = self.model(
                input_ids=prefix_input_ids[:, start:end],
                attention_mask=prefix_attention_mask[:, :end],
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
            )
            past = outputs.past_key_values

        return outputs

    def _build_initial_store(self, past_key_values):
        full_past, tail_past, _ = split_full_blocks_and_tail(
            past_key_values,
            tokens_per_block=self.tokens_per_block,
        )
        kv_blocks = real_past_to_blocks(
            full_past, tokens_per_block=self.tokens_per_block
        )

        store = KVBlockStore(tokens_per_block=self.tokens_per_block)
        for block_id, (key, value) in enumerate(kv_blocks):
            store.put_gpu(block_id, key, value)

        return store, len(full_past), len(kv_blocks), tail_past

    def _append_new_full_blocks(
            self,
            *,
            store: KVBlockStore,
            past_key_values,
            known_num_blocks: int,
    ):
        full_past, tail_past, full_tokens = split_full_blocks_and_tail(
            past_key_values,
            tokens_per_block=self.tokens_per_block,
        )
        new_num_blocks = full_tokens // self.tokens_per_block
        added_block_ids: list[int] = []

        if new_num_blocks > known_num_blocks:
            for block_id in range(known_num_blocks, new_num_blocks):
                key, value = extract_single_block_from_past(
                    full_past,
                    block_id=block_id,
                    tokens_per_block=self.tokens_per_block,
                )
                store.put_gpu(block_id, key, value)
                added_block_ids.append(block_id)

        return new_num_blocks, tail_past, added_block_ids

    @staticmethod
    def _reload_all_blocks(
            store: KVBlockStore,
            device: torch.device,
            num_blocks: int,
    ) -> tuple[int, int]:
        before = store.summary()
        for block_id in range(num_blocks):
            store.ensure_gpu(block_id, device)
        after = store.summary()

        moved_bytes = after["cpu_to_gpu_bytes"] - before["cpu_to_gpu_bytes"]
        moved_copies = after["cpu_to_gpu_copies"] - before["cpu_to_gpu_copies"]
        return moved_bytes, moved_copies
