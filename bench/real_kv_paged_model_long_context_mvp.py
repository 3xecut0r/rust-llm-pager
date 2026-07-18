from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, TOKENS_PER_BLOCK, VRAM_BUDGET, RAM_BUDGET, RECENT_WINDOW
from pager_hf import PagedModel

# recent_only doesn't need attention scores for placement, so PagedModel
# skips output_attentions entirely and stays on the chunked-prefill path,
# which is what actually makes a long context tractable. See the "Does
# this actually save VRAM?" section in README for why heavy_hitter-style
# policies can't do this today.
POLICY = "recent_only"

CONTEXT_TOKENS = 6000
NEW_TOKENS = 8
PREFILL_CHUNK = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate pager_hf.PagedModel itself at a long context (not just the low-level primitives)."
    )
    parser.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    return parser.parse_args()


def build_long_prompt(tokenizer, target_tokens: int) -> str:
    paragraph = (
        "The quick brown fox jumps over the lazy dog while researchers discuss "
        "memory management, operating systems, distributed caches, and GPU "
        "scheduling in long, unrelated technical documents. "
    )
    paragraph_tokens = len(tokenizer(paragraph)["input_ids"])
    repeats = target_tokens // max(paragraph_tokens, 1) + 4
    return paragraph * repeats


def chunked_baseline_generate(
        *,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        new_tokens: int,
        chunk_size: int,
) -> list[int]:
    prefix_input_ids = input_ids[:, :-1]
    prefix_attention_mask = attention_mask[:, :-1]
    seq_len = prefix_input_ids.shape[-1]

    past = None
    outputs = None

    with torch.inference_mode():
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            outputs = model(
                input_ids=prefix_input_ids[:, start:end],
                attention_mask=prefix_attention_mask[:, :end],
                past_key_values=past,
                use_cache=True,
                output_attentions=False,
            )
            past = outputs.past_key_values

    next_input_id = input_ids[:, -1:]
    current_mask = attention_mask
    current_cache = outputs.past_key_values
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(new_tokens):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_mask,
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )
            next_token_id = torch.argmax(
                outputs.logits[:, -1, :], dim=-1, keepdim=True
            )
            generated.append(int(next_token_id.item()))

            current_cache = outputs.past_key_values
            next_input_id = next_token_id
            current_mask = torch.cat(
                [
                    current_mask,
                    torch.ones(
                        (current_mask.shape[0], 1),
                        dtype=current_mask.dtype,
                        device=current_mask.device,
                    ),
                ],
                dim=1,
            )

    return generated


def main() -> None:
    args = parse_args()
    context_tokens = args.context_tokens
    new_tokens = args.new_tokens

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", POLICY)
    print("context_tokens (target):", context_tokens)
    print("new_tokens:", new_tokens)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    prompt = build_long_prompt(tokenizer, context_tokens)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=context_tokens,
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("actual_context_tokens:", input_ids.shape[-1])

    print("\nRunning baseline (chunked prefill, full KV resident, no paging)...")
    baseline_generated = chunked_baseline_generate(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        new_tokens=new_tokens,
        chunk_size=PREFILL_CHUNK,
    )

    print("Running PagedModel.generate() at the same long context...")
    paged_model = PagedModel(
        model,
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        recent_window=RECENT_WINDOW,
        policy=POLICY,
        tokens_per_block=TOKENS_PER_BLOCK,
        prefill_chunk_tokens=PREFILL_CHUNK,
    )
    paged_generated = paged_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=new_tokens,
    )

    stats = paged_model.last_run_stats

    print("\nPagedModel long-context summary")
    print("--------------------------------")
    print("baseline_ids:", baseline_generated)
    print("paged_ids:", paged_generated)
    print("same_token_ids:", baseline_generated == paged_generated)
    print("total_new_blocks:", stats.total_new_blocks)
    print("final_num_blocks:", stats.final_num_blocks)
    print("total_gpu_to_cpu_mb:", f"{stats.total_gpu_to_cpu_mb:.2f}")
    print("total_cpu_to_gpu_mb:", f"{stats.total_cpu_to_gpu_mb:.2f}")

    assert baseline_generated == paged_generated

    print(
        "\nOK: pager_hf.PagedModel handles a long context directly, "
        "with byte-identical output to baseline."
    )


if __name__ == "__main__":
    main()
