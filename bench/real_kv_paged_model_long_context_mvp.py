from __future__ import annotations

import argparse

import torch
from config import MODEL_NAME, RAM_BUDGET, RECENT_WINDOW, TOKENS_PER_BLOCK, VRAM_BUDGET
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel

# Every PagedModel forward call processes exactly one token at a time
# (both priming and decoding go through _forward_step, never a multi-token
# chunk), so requesting output_attentions=True there costs a *linear*
# [heads, 1, total_len] eager-attention matrix per layer, not the quadratic
# [heads, seq_len, seq_len] a bulk forward call would need. That's why
# attention-scored policies (heavy_hitter, sinks_heavy_hitter) also scale to
# long context, not just recent_only/sinks_recent — pass --policy to try
# either kind.
DEFAULT_POLICY = "recent_only"

CONTEXT_TOKENS = 6000
NEW_TOKENS = 8
PREFILL_CHUNK = 512


def parse_args() -> argparse.Namespace:
    """Parse --context-tokens, --new-tokens, and --policy."""
    parser = argparse.ArgumentParser(
        description="Validate pager_hf.PagedModel itself at a long context (not just the low-level primitives)."
    )
    parser.add_argument("--context-tokens", type=int, default=CONTEXT_TOKENS)
    parser.add_argument("--new-tokens", type=int, default=NEW_TOKENS)
    parser.add_argument(
        "--policy",
        default=DEFAULT_POLICY,
        choices=["recent_only", "sinks_recent", "heavy_hitter", "sinks_heavy_hitter"],
        help="pager placement policy to validate at long context (default: %(default)s)",
    )
    return parser.parse_args()


def build_long_prompt(tokenizer, target_tokens: int) -> str:
    """Repeat a filler paragraph until it tokenizes to at least target_tokens."""
    paragraph = (
        "The quick brown fox jumps over the lazy dog while researchers discuss "
        "memory management, operating systems, distributed caches, and GPU "
        "scheduling in long, unrelated technical documents. "
    )
    paragraph_tokens = len(tokenizer(paragraph)["input_ids"])
    repeats = target_tokens // max(paragraph_tokens, 1) + 4
    return paragraph * repeats


def chunked_baseline_generate(
    *, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, new_tokens: int, chunk_size: int
) -> list[int]:
    """Generate new_tokens the plain way, prefilling in chunks so long contexts don't OOM on logits."""
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


def main() -> None:
    args = parse_args()
    context_tokens = args.context_tokens
    new_tokens = args.new_tokens
    policy = args.policy

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", policy)
    print("context_tokens (target):", context_tokens)
    print("new_tokens:", new_tokens)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()

    prompt = build_long_prompt(tokenizer, context_tokens)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=context_tokens)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("actual_context_tokens:", input_ids.shape[-1])

    print("\nRunning baseline (chunked prefill, full KV resident, no paging)...")
    baseline_generated = chunked_baseline_generate(
        model=model, input_ids=input_ids, attention_mask=attention_mask, new_tokens=new_tokens, chunk_size=PREFILL_CHUNK
    )

    print("Running PagedModel.generate() at the same long context...")
    torch.cuda.reset_peak_memory_stats(device)
    paged_model = PagedModel(
        model,
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        recent_window=RECENT_WINDOW,
        policy=policy,
        tokens_per_block=TOKENS_PER_BLOCK,
        prefill_chunk_tokens=PREFILL_CHUNK,
    )
    paged_generated = paged_model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=new_tokens
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
    print("mean_attention_in_gpu:", f"{stats.mean_attention_in_gpu:.4f}")
    print("min_attention_in_gpu:", f"{stats.min_attention_in_gpu:.4f}")
    print("max_attention_in_gpu:", f"{stats.max_attention_in_gpu:.4f}")
    print("peak_gpu_mb:", f"{torch.cuda.max_memory_allocated(device) / 1_000_000:.2f}")

    assert baseline_generated == paged_generated

    print("\nOK: pager_hf.PagedModel handles a long context directly, with byte-identical output to baseline.")


if __name__ == "__main__":
    main()
