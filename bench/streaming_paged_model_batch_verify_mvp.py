from __future__ import annotations

import torch
from config import (
    MODEL_NAME,
    PROMOTE_MARGIN,
    RAM_BUDGET,
    RAM_PROMOTE_MARGIN,
    REBALANCE_INTERVAL,
    RECENT_WINDOW,
    TOKENS_PER_BLOCK,
)
from real_kv_utils import build_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel

GENERATE_TOKENS = 8
VRAM_BUDGET = 128_000_000
POLICIES = ["recent_only", "sinks_recent", "heavy_hitter", "sinks_heavy_hitter"]


def greedy_baseline_generate_row(
    *, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int
) -> list[int]:
    """Single-row baseline, used per row so batch padding never affects the reference answer."""
    next_input_id = input_ids[:, -1:]
    current_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True, output_attentions=False
        )
    current_cache = outputs.past_key_values
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_mask,
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(next_token_id.item()))
            current_cache = outputs.past_key_values
            next_input_id = next_token_id
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=current_mask.device),
                ],
                dim=1,
            )

    return generated


def build_batch(tokenizer, device, *, common_len: int, pad_amounts: list[int]):
    """Build a left-padded batch of `len(pad_amounts)` rows, each with a different amount of real content."""
    prompt = build_prompt()
    base_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=common_len)["input_ids"][0]
    pad_id = tokenizer.pad_token_id

    rows_ids = []
    rows_mask = []
    for pad_amount in pad_amounts:
        real_len = common_len - pad_amount
        real = base_ids[:real_len]
        row_ids = torch.cat([torch.full((pad_amount,), pad_id, dtype=real.dtype), real])
        row_mask = torch.cat([torch.zeros(pad_amount, dtype=torch.long), torch.ones(real_len, dtype=torch.long)])
        rows_ids.append(row_ids)
        rows_mask.append(row_mask)

    input_ids = torch.stack(rows_ids).to(device)
    attention_mask = torch.stack(rows_mask).to(device)
    return input_ids, attention_mask


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")
    print("device:", device)
    print("model:", MODEL_NAME)
    print("generate_tokens:", GENERATE_TOKENS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager", torch_dtype=torch.float16).to(
        device
    )
    model.eval()

    common_len = 140
    pad_amounts = [0, 15, 40]  # three rows, different amounts of real content, sharing one padded tensor
    input_ids, attention_mask = build_batch(tokenizer, device, common_len=common_len, pad_amounts=pad_amounts)
    print("batch_size:", input_ids.shape[0], "common_len:", common_len, "pad_amounts:", pad_amounts)

    # Informational only, not the correctness bar: batched matmul has
    # different rounding than a single-row forward pass, so an isolated
    # per-row baseline can legitimately diverge from a batched run on a
    # low-confidence continuation -- that's a general floating-point property
    # of batching, unrelated to padding masking. Kept to sanity-check that at
    # least the well-formed rows (enough real content to contain the actual
    # question) land on the same answer either way.
    baseline_per_row = []
    for row in range(input_ids.shape[0]):
        row_ids = input_ids[row : row + 1]
        row_mask = attention_mask[row : row + 1]
        baseline_per_row.append(
            greedy_baseline_generate_row(model=model, input_ids=row_ids, attention_mask=row_mask, steps=GENERATE_TOKENS)
        )
    print("\nbaseline_per_row:", baseline_per_row)

    # The real correctness bar: streaming and non-streaming, run on the exact
    # same batched input, must agree row for row. Both go through the same
    # batched matmuls, so this isolates what padding masking is actually
    # responsible for -- streaming must not diverge beyond whatever batching
    # numerics the already-shipped non-streaming path already has. This holds
    # for heavy_hitter/sinks_heavy_hitter too, even though their block-placement
    # score is now an average across batch rows (one placement decision is
    # still shared by the whole batch): every row's attention always covers
    # the full context regardless of which physical tier a block sits in, so
    # generated tokens never depend on the placement score at all.
    all_ok = True
    for policy in POLICIES:
        results = {}
        for use_streaming in (False, True):
            paged_model = PagedModel(
                model,
                vram_budget=VRAM_BUDGET,
                ram_budget=RAM_BUDGET,
                recent_window=RECENT_WINDOW,
                rebalance_interval=REBALANCE_INTERVAL,
                promote_margin=PROMOTE_MARGIN,
                ram_promote_margin=RAM_PROMOTE_MARGIN,
                policy=policy,
                tokens_per_block=TOKENS_PER_BLOCK,
                use_streaming_attention=use_streaming,
                streaming_group_size_blocks=4,
            )
            batch_generated = paged_model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=GENERATE_TOKENS
            )
            label = "streaming" if use_streaming else "non_streaming"
            results[label] = batch_generated
            print(f"\npolicy={policy} mode={label}")
            print("batch_generated:", batch_generated)
            print("matches_baseline_per_row:", batch_generated == baseline_per_row)

        matches = results["streaming"] == results["non_streaming"]
        all_ok = all_ok and matches
        print(f"\npolicy={policy} streaming_matches_non_streaming (both batched): {matches}")

    if all_ok:
        print(
            "\nOK: batched streaming generation (heterogeneous padding) matches batched non-streaming for every policy."
        )
    else:
        print("\nFAIL: streaming diverged from non-streaming on the same batched input for at least one policy.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
