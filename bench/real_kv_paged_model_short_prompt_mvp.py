from __future__ import annotations

import torch
from config import MODEL_NAME, RAM_BUDGET, VRAM_BUDGET
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel

# Regression proof for a real bug: a prompt shorter than one tokens_per_block
# left _num_blocks at 0 for the first several decode steps (everything still
# in the tail, nothing promoted to a full pager block yet). Every policy hit
# this, just with a different exception -- ZeroDivisionError for
# recent_only/sinks_recent (a uniform 1/num_blocks vector), OverflowError for
# heavy_hitter/sinks_heavy_hitter (query_block = num_blocks - 1 went
# negative, rejected by the Rust pager's u64 argument), and a separate
# KeyError on the non-streaming path specifically (reconstruct_past_from_store
# assumed block 0 always existed to sample a shape from). Found while
# building pager_hf.serving; fixed in paged_model.py/_forward_step and
# kv_utils.py/reconstruct_past_from_store, not in the serving layer itself.

POLICIES = ["recent_only", "sinks_recent", "heavy_hitter", "sinks_heavy_hitter"]
PROMPT = "The quick brown fox jumps over the lazy dog. Once upon a time,"  # well under tokens_per_block=16
GENERATE_TOKENS = 8


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()

    encoded = tokenizer(PROMPT, return_tensors="pt").to(device)
    print("prompt_tokens:", encoded["input_ids"].shape[-1], "(tokens_per_block default is 16)")

    baseline_ids: list[int] = []
    ids, mask = encoded["input_ids"], encoded["attention_mask"]
    with torch.inference_mode():
        for _ in range(GENERATE_TOKENS):
            out = model(input_ids=ids, attention_mask=mask, use_cache=True)
            next_id = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
            baseline_ids.append(int(next_id.item()))
            ids = torch.cat([ids, next_id], dim=1)
            mask = torch.cat([mask, torch.ones((1, 1), dtype=mask.dtype, device=mask.device)], dim=1)
    print("baseline_ids:", baseline_ids)

    all_ok = True
    for policy in POLICIES:
        for use_streaming in (True, False):
            label = f"policy={policy} streaming={use_streaming}"
            try:
                paged_model = PagedModel(
                    model,
                    vram_budget=VRAM_BUDGET,
                    ram_budget=RAM_BUDGET,
                    policy=policy,
                    use_streaming_attention=use_streaming,
                )
                generated = paged_model.generate(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    max_new_tokens=GENERATE_TOKENS,
                )
                matches = generated == baseline_ids
                print(f"{label}: {'OK' if matches else 'MISMATCH'} {generated}")
                all_ok = all_ok and matches
            except Exception as exc:  # noqa: BLE001 -- deliberately catching everything to report all 8 combos
                print(f"{label}: FAILED -- {type(exc).__name__}: {exc}")
                all_ok = False

    assert all_ok, "at least one policy/streaming combination diverged or crashed on a short prompt"
    print("\nOK: every policy, streaming and non-streaming, handles a short (sub-tokens_per_block) prompt correctly.")


if __name__ == "__main__":
    main()
