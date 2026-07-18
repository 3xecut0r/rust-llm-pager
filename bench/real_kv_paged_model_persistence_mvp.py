from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_NAME, TOKENS_PER_BLOCK, VRAM_BUDGET, RAM_BUDGET, RECENT_WINDOW
from real_kv_utils import build_prompt
from pager_hf import PagedModel

POLICY = "recent_only"
GENERATE_TOKENS = 32


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", POLICY)
    print("generate_tokens:", GENERATE_TOKENS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
    ).to(device)
    model.eval()

    prompt = build_prompt()
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    print("prompt_seq_len:", input_ids.shape[-1])

    pager_kwargs = dict(
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        recent_window=RECENT_WINDOW,
        policy=POLICY,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    # ---- Single call: the whole prompt primed and generated in one shot ----
    print("\nRunning single-call generate() (one shot, no persistence needed)...")
    single_call_model = PagedModel(model, **pager_kwargs)
    single_call_generated = single_call_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=GENERATE_TOKENS,
    )

    # ---- Split across two calls: prime half the prompt first (no
    # generation), then continue with the full prompt and actually
    # generate. If persistence is correct, this must produce exactly the
    # same tokens as the single call above. ----
    print("Running split generate() calls (persistent session, primed in two steps)...")
    split_point = input_ids.shape[-1] // 2

    persistent_model = PagedModel(model, **pager_kwargs)
    persistent_model.generate(
        input_ids=input_ids[:, :split_point],
        attention_mask=attention_mask[:, :split_point],
        max_new_tokens=0,
    )
    split_call_generated = persistent_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=GENERATE_TOKENS,
    )

    print("\nPersistence summary")
    print("--------------------")
    print("single_call_ids:", single_call_generated)
    print("split_call_ids:", split_call_generated)
    print("same_token_ids:", single_call_generated == split_call_generated)

    assert single_call_generated == split_call_generated

    # ---- Safety checks: reset() and prefix-mismatch/rewind must be caught ----
    print("\nChecking reset() and misuse guards...")
    persistent_model.reset()
    persistent_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
    )
    print("reset() + fresh generate(): OK")

    diverged_model = PagedModel(model, **pager_kwargs)
    diverged_model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=4,
    )
    try:
        diverged_model.generate(
            input_ids=input_ids[:, :5],
            attention_mask=attention_mask[:, :5],
            max_new_tokens=1,
        )
    except ValueError as exc:
        print("rewind correctly rejected:", str(exc)[:80])
    else:
        raise AssertionError("expected ValueError for a rewound input_ids")

    print(
        "\nOK: PagedModel persists correctly across generate() calls, "
        "with identical output to a single-call run and safe misuse guards."
    )


if __name__ == "__main__":
    main()
