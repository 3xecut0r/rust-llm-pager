from __future__ import annotations

import torch
from config import MODEL_NAME, RAM_BUDGET, VRAM_BUDGET
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel

# Correctness bar for beam search is agreement with HuggingFace's own
# model.generate(num_beams=...), not same_token_ids against greedy -- beam
# search is *expected* to sometimes diverge from greedy, that's the point of
# it. Covers the follow-ups added on top of the original single-prompt V1:
# length_penalty (both 0.0, the original no-normalization behavior, and the
# new default 1.0 matching HuggingFace's own), num_return_sequences > 1, and
# a batch of independent prompts. do_sample=True is checked only for basic
# sanity (valid, non-crashing output) -- HuggingFace's own sampling RNG
# doesn't match this implementation's, so exact agreement isn't the bar there.

NUM_BEAMS = 4
GENERATE_TOKENS = 12
PROMPTS = ["The capital of France is", "Water boils at a temperature of"]


def hf_beam_search(model, tokenizer, input_ids, attention_mask, *, length_penalty, num_return_sequences=1):
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        num_beams=NUM_BEAMS,
        max_new_tokens=GENERATE_TOKENS,
        do_sample=False,
        length_penalty=length_penalty,
        early_stopping=False,
        num_return_sequences=num_return_sequences,
        pad_token_id=tokenizer.pad_token_id,
    )
    prompt_len = input_ids.shape[-1]
    flat_seqs = output[:, prompt_len:].tolist()
    batch_size = input_ids.shape[0]
    # HF groups rows [prompt0_seq0, prompt0_seq1, ..., prompt1_seq0, ...] when num_return_sequences > 1.
    return [flat_seqs[i * num_return_sequences : (i + 1) * num_return_sequences] for i in range(batch_size)]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")
    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("num_beams:", NUM_BEAMS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()

    single = tokenizer(PROMPTS[0], return_tensors="pt").to(device)
    batched = tokenizer(PROMPTS, return_tensors="pt", padding=True).to(device)

    all_ok = True

    print("\n--- Test A: single prompt, length_penalty=0.0 (original no-normalization behavior) ---")
    hf_ids = hf_beam_search(model, tokenizer, single["input_ids"], single["attention_mask"], length_penalty=0.0)[0][0]
    for use_streaming in (True, False):
        paged_model = PagedModel(
            model,
            vram_budget=VRAM_BUDGET,
            ram_budget=RAM_BUDGET,
            policy="sinks_heavy_hitter",
            use_streaming_attention=use_streaming,
        )
        paged_ids = paged_model.generate_beam_search(
            input_ids=single["input_ids"],
            attention_mask=single["attention_mask"],
            num_beams=NUM_BEAMS,
            max_new_tokens=GENERATE_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            length_penalty=0.0,
        )
        matches = paged_ids == hf_ids
        print(f"use_streaming_attention={use_streaming}: matches_hf={matches}")
        all_ok = all_ok and matches

    print("\n--- Test B: single prompt, length_penalty=1.0 (new default, matches HF's own default) ---")
    hf_ids = hf_beam_search(model, tokenizer, single["input_ids"], single["attention_mask"], length_penalty=1.0)[0][0]
    paged_model = PagedModel(model, vram_budget=VRAM_BUDGET, ram_budget=RAM_BUDGET, policy="sinks_heavy_hitter")
    paged_ids = paged_model.generate_beam_search(
        input_ids=single["input_ids"],
        attention_mask=single["attention_mask"],
        num_beams=NUM_BEAMS,
        max_new_tokens=GENERATE_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
    )
    matches = paged_ids == hf_ids
    print(f"default length_penalty: matches_hf={matches}")
    all_ok = all_ok and matches

    print("\n--- Test C: single prompt, num_return_sequences=2 ---")
    hf_ids = hf_beam_search(
        model, tokenizer, single["input_ids"], single["attention_mask"], length_penalty=1.0, num_return_sequences=2
    )[0]
    paged_model = PagedModel(model, vram_budget=VRAM_BUDGET, ram_budget=RAM_BUDGET, policy="sinks_heavy_hitter")
    paged_ids = paged_model.generate_beam_search(
        input_ids=single["input_ids"],
        attention_mask=single["attention_mask"],
        num_beams=NUM_BEAMS,
        max_new_tokens=GENERATE_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=2,
    )
    matches = paged_ids == hf_ids
    print(f"num_return_sequences=2: paged={paged_ids} hf={hf_ids} matches_hf={matches}")
    all_ok = all_ok and matches

    print("\n--- Test D: batch of 2 independent prompts ---")
    hf_ids = hf_beam_search(model, tokenizer, batched["input_ids"], batched["attention_mask"], length_penalty=1.0)
    hf_ids = [group[0] for group in hf_ids]
    paged_model = PagedModel(model, vram_budget=VRAM_BUDGET, ram_budget=RAM_BUDGET, policy="sinks_heavy_hitter")
    paged_ids = paged_model.generate_beam_search(
        input_ids=batched["input_ids"],
        attention_mask=batched["attention_mask"],
        num_beams=NUM_BEAMS,
        max_new_tokens=GENERATE_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
    )
    matches = paged_ids == hf_ids
    print(f"batch_size=2: paged={paged_ids} hf={hf_ids} matches_hf={matches}")
    all_ok = all_ok and matches

    print("\n--- Test E: do_sample=True (sanity only, not exact-match -- RNG differs from HF's) ---")
    paged_model = PagedModel(model, vram_budget=VRAM_BUDGET, ram_budget=RAM_BUDGET, policy="sinks_heavy_hitter")
    sampled_ids = paged_model.generate_beam_search(
        input_ids=single["input_ids"],
        attention_mask=single["attention_mask"],
        num_beams=NUM_BEAMS,
        max_new_tokens=GENERATE_TOKENS,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        generator=torch.Generator(device=device).manual_seed(0),
    )
    valid = isinstance(sampled_ids, list) and all(
        isinstance(t, int) and 0 <= t < model.config.vocab_size for t in sampled_ids
    )
    print(f"do_sample output: {sampled_ids} valid={valid}")
    all_ok = all_ok and valid

    assert all_ok, "at least one beam-search check failed"
    print(
        "\nOK: generate_beam_search matches HuggingFace's own beam search across length_penalty, num_return_sequences, and batch_size, plus a do_sample sanity check."
    )


if __name__ == "__main__":
    main()
