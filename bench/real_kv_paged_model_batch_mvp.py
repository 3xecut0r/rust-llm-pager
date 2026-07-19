from __future__ import annotations

import torch
from config import MODEL_NAME, RAM_BUDGET, RECENT_WINDOW, TOKENS_PER_BLOCK, VRAM_BUDGET
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel

POLICY = "recent_only"
NEW_TOKENS = 16

PROMPTS = [
    (
        "IMPORTANT FACT: The secret project codename is BLUE ORCHID. "
        "Remember this codename because it will be asked later. "
        "This paragraph is unrelated filler text about software engineering, "
        "memory management, operating systems, compilers, databases, networking, "
        "and performance optimization. It mentions Rust, Python, Linux, GPUs, "
        "caches, filesystems, and distributed systems, but it does not contain "
        "the secret project codename. "
        "Question: What is the secret project codename mentioned at the beginning?\n"
        "Answer:"
    ),
    (
        "IMPORTANT FACT: The launch sequence password is CRIMSON FALCON. "
        "Remember this password because it will be asked later. "
        "This paragraph is unrelated filler text about cooking, gardening, "
        "travel, weather patterns, historical events, and sports statistics. "
        "It mentions Paris, mountains, rivers, and festivals, but it does not "
        "contain the launch sequence password. "
        "Question: What is the launch sequence password mentioned at the beginning?\n"
        "Answer:"
    ),
    (
        "IMPORTANT FACT: The vault access code is SILVER MERIDIAN. "
        "Remember this code because it will be asked later. "
        "This paragraph is unrelated filler text about music theory, painting "
        "techniques, ancient philosophy, chemistry experiments, and astronomy. "
        "It mentions stars, pigments, scales, and reactions, but it does not "
        "contain the vault access code. "
        "Question: What is the vault access code mentioned at the beginning?\n"
        "Answer:"
    ),
]


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")
    batch_size = len(PROMPTS)

    print("device:", device)
    print("model:", MODEL_NAME)
    print("policy:", POLICY)
    print("batch_size:", batch_size)
    print("new_tokens:", NEW_TOKENS)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).to(device)
    model.eval()

    # Every row in a batch must have the same *total* length. Rows are
    # naturally different lengths here, so left-pad the shorter ones up to
    # the longest, matching the standard convention for batched causal-LM
    # generation (right-padding would break "next token" alignment).
    pad_token_id = tokenizer.pad_token_id
    unpadded_rows = [tokenizer(prompt, return_tensors="pt")["input_ids"] for prompt in PROMPTS]
    context_tokens = max(ids.shape[-1] for ids in unpadded_rows)

    padded_rows = []
    masks = []
    for ids in unpadded_rows:
        pad_amount = context_tokens - ids.shape[-1]
        pad = torch.full((1, pad_amount), pad_token_id, dtype=ids.dtype)
        padded_rows.append(torch.cat([pad, ids], dim=1))
        masks.append(
            torch.cat(
                [torch.zeros(1, pad_amount, dtype=torch.long), torch.ones(1, ids.shape[-1], dtype=torch.long)], dim=1
            )
        )

    input_ids = torch.cat(padded_rows, dim=0).to(device)
    attention_mask = torch.cat(masks, dim=0).to(device)

    print("row lengths (unpadded):", [ids.shape[-1] for ids in unpadded_rows])
    print("context_tokens (padded to longest prompt):", context_tokens)
    print("batched_input_shape:", tuple(input_ids.shape))

    pager_kwargs = dict(
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        recent_window=RECENT_WINDOW,
        policy=POLICY,
        tokens_per_block=TOKENS_PER_BLOCK,
    )

    print("\nRunning batch_size=1 (one fresh PagedModel per row, no padding) for reference...")
    per_row_generated = []
    for row in range(batch_size):
        row_model = PagedModel(model, **pager_kwargs)
        row_ids = unpadded_rows[row].to(device)
        row_generated = row_model.generate(
            input_ids=row_ids, attention_mask=torch.ones_like(row_ids), max_new_tokens=NEW_TOKENS
        )
        per_row_generated.append(row_generated)

    print("Running one fused, left-padded batch_size=%d generate() call..." % batch_size)
    batch_model = PagedModel(model, **pager_kwargs)
    batch_generated = batch_model.generate(
        input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=NEW_TOKENS
    )

    print("\nBatch summary")
    print("-------------")
    for row in range(batch_size):
        print(f"row {row} per_row_ids (unpadded):", per_row_generated[row])
        print(f"row {row} batch_ids (left-padded):", batch_generated[row])
        print(f"row {row} same:                   ", per_row_generated[row] == batch_generated[row])

    assert isinstance(batch_generated, list) and len(batch_generated) == batch_size
    for row in range(batch_size):
        assert (
            per_row_generated[row] == batch_generated[row]
        ), f"row {row}: left-padded batch output diverged from the unpadded batch_size=1 reference"

    stats = batch_model.last_run_stats
    print("\nfinal_num_blocks:", stats.final_num_blocks)
    print("total_new_blocks:", stats.total_new_blocks)

    print("\nChecking guardrails...")

    try:
        heavy_model = PagedModel(model, **{**pager_kwargs, "policy": "sinks_heavy_hitter"})
        heavy_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=1)
    except NotImplementedError as exc:
        print("attention-needing policy + batch>1 correctly rejected:", str(exc)[:80])
    else:
        raise AssertionError("expected NotImplementedError for heavy_hitter at batch>1")

    try:
        right_padded_mask = attention_mask.clone()
        right_padded_mask[0, -1] = 0  # trailing zero: not left-padding
        pad_model = PagedModel(model, **pager_kwargs)
        pad_model.generate(input_ids=input_ids, attention_mask=right_padded_mask, max_new_tokens=1)
    except NotImplementedError as exc:
        print("right-padded attention_mask correctly rejected:", str(exc)[:80])
    else:
        raise AssertionError("expected NotImplementedError for a right-padded attention_mask")

    print(
        "\nOK: batched PagedModel.generate() with left-padding matches unpadded "
        "batch_size=1 generation exactly, and unsupported configurations are "
        "rejected explicitly."
    )


if __name__ == "__main__":
    main()
