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

    Greedy decoding only for now. Each `generate()` call starts a fresh
    KV block store and pager; the persistent-across-steps part is the
    generation loop within a single call.
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
    ):
        self.model = model
        self.tokens_per_block = tokens_per_block

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

    @torch.inference_mode()
    def generate(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            max_new_tokens: int,
    ) -> list[int]:
        device = input_ids.device

        prefix_input_ids = input_ids[:, :-1]
        prefix_attention_mask = attention_mask[:, :-1]

        next_input_id = input_ids[:, -1:]
        current_attention_mask = attention_mask

        outputs = self.model(
            input_ids=prefix_input_ids,
            attention_mask=prefix_attention_mask,
            use_cache=True,
            output_attentions=True,
        )

        current_past = get_legacy_past_key_values(outputs)

        store, num_layers, num_blocks, tail_past = self._build_initial_store(
            current_past
        )

        rust_pager = _pager_rs.PyPager(**self._pager_kwargs)

        generated: list[int] = []
        stats = GenerationStats()
        attention_in_gpu_values: list[float] = []

        for _ in range(max_new_tokens):
            reload_bytes, reload_copies = self._reload_all_blocks(
                store, device, num_blocks
            )

            reconstructed_full_past = reconstruct_past_from_store(
                store=store,
                num_layers=num_layers,
                num_blocks=num_blocks,
            )
            current_past_for_forward = append_tail_to_reconstructed_past(
                reconstructed_full_past,
                tail_past,
            )
            cache = DynamicCache.from_legacy_cache(tuple(current_past_for_forward))

            outputs = self.model(
                input_ids=next_input_id,
                attention_mask=current_attention_mask,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
            )

            next_token_id = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True
            )
            generated.append(int(next_token_id.item()))

            current_past = get_legacy_past_key_values(outputs)

            old_num_blocks = num_blocks
            num_blocks, tail_past, added_block_ids = self._append_new_full_blocks(
                store=store,
                past_key_values=current_past,
                known_num_blocks=old_num_blocks,
            )
            stats.total_new_blocks += len(added_block_ids)

            block_attention = extract_last_query_block_attention(
                outputs,
                tokens_per_block=self.tokens_per_block,
                num_blocks=num_blocks,
            )
            query_block = num_blocks - 1

            summary_before = store.summary()
            rust_pager.on_step(query_block, 0, block_attention)
            if added_block_ids:
                rust_pager.force_rebalance(query_block)
            store.apply_tiers(rust_pager.tiers(), device)
            summary_after = store.summary()

            stats.total_gpu_to_cpu_mb += (
                summary_after["gpu_to_cpu_bytes"] - summary_before["gpu_to_cpu_bytes"]
            ) / 1_000_000
            stats.total_cpu_to_gpu_mb += reload_bytes / 1_000_000
            stats.total_gpu_to_cpu_copies += (
                summary_after["gpu_to_cpu_copies"]
                - summary_before["gpu_to_cpu_copies"]
            )
            stats.total_cpu_to_gpu_copies += reload_copies

            attention_in_gpu = sum(
                block_attention[block_id]
                for block_id in store.gpu_block_ids()
                if block_id < len(block_attention)
            )
            attention_in_gpu_values.append(attention_in_gpu)

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

        stats.final_num_blocks = num_blocks

        if attention_in_gpu_values:
            stats.mean_attention_in_gpu = sum(attention_in_gpu_values) / len(
                attention_in_gpu_values
            )
            stats.min_attention_in_gpu = min(attention_in_gpu_values)
            stats.max_attention_in_gpu = max(attention_in_gpu_values)

        self.last_run_stats = stats

        return generated

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
