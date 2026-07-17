from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import (
    MODEL_NAME,
    MAX_LENGTH,
    VRAM_BUDGET,
    RAM_BUDGET,
    RECENT_WINDOW,
    REBALANCE_INTERVAL,
    PROMOTE_MARGIN,
    RAM_PROMOTE_MARGIN,
    POLICY,
    TOKENS_PER_BLOCK,
)
from real_kv_utils import build_prompt
from pager_hf import PagedModel

GENERATE_TOKENS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the pager_hf.PagedModel API against baseline greedy generation"
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=GENERATE_TOKENS,
        help="number of tokens to generate (default: %(default)s)",
    )
    return parser.parse_args()


def greedy_baseline_generate(
        *,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        steps: int,
) -> list[int]:
    prefix_input_ids = input_ids[:, :-1]
    prefix_attention_mask = attention_mask[:, :-1]

    next_input_id = input_ids[:, -1:]
    current_attention_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=prefix_input_ids,
            attention_mask=prefix_attention_mask,
            use_cache=True,
            output_attentions=False,
        )

    current_cache = outputs.past_key_values
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_attention_mask,
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )

            next_token_id = torch.argmax(
                outputs.logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )

            generated.append(int(next_token_id.item()))

            current_cache = outputs.past_key_values
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

    return generated


def main() -> None:
    args = parse_args()
    generate_tokens = args.tokens

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", POLICY)
    print("generate_tokens:", generate_tokens)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    ).to(device)

    model.eval()

    prompt = build_prompt()

    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("prompt_seq_len:", input_ids.shape[-1])

    baseline_generated = greedy_baseline_generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        steps=generate_tokens,
    )

    paged_model = PagedModel(
        model,
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        recent_window=RECENT_WINDOW,
        rebalance_interval=REBALANCE_INTERVAL,
        promote_margin=PROMOTE_MARGIN,
        ram_promote_margin=RAM_PROMOTE_MARGIN,
        policy=POLICY,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    paged_generated = paged_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=generate_tokens,
    )

    stats = paged_model.last_run_stats

    baseline_text = tokenizer.decode(baseline_generated)
    paged_text = tokenizer.decode(paged_generated)

    print("\nPagedModel API summary")
    print("-----------------------")
    print("baseline_ids:", baseline_generated)
    print("paged_ids:", paged_generated)
    print("same_token_ids:", baseline_generated == paged_generated)
    print("baseline_text:", repr(baseline_text))
    print("paged_text:", repr(paged_text))
    print("total_new_blocks:", stats.total_new_blocks)
    print("final_num_blocks:", stats.final_num_blocks)
    print("total_gpu_to_cpu_mb:", f"{stats.total_gpu_to_cpu_mb:.2f}")
    print("total_cpu_to_gpu_mb:", f"{stats.total_cpu_to_gpu_mb:.2f}")
    print("mean_attention_in_gpu:", f"{stats.mean_attention_in_gpu:.4f}")
    print("min_attention_in_gpu:", f"{stats.min_attention_in_gpu:.4f}")
    print("max_attention_in_gpu:", f"{stats.max_attention_in_gpu:.4f}")

    assert baseline_generated == paged_generated

    print(
        "\nOK: pager_hf.PagedModel produces identical greedy generation to baseline."
    )


if __name__ == "__main__":
    main()
