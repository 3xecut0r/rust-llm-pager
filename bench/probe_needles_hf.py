from __future__ import annotations

from textwrap import dedent

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_LENGTH = 256
MAX_NEW_TOKENS = 8


CANDIDATES = [
    {
        "needle_id": "blue_orchid",
        "answer": "BLUE ORCHID",
        "fact": "The secret project codename is BLUE ORCHID.",
        "question": "What is the secret project codename mentioned at the beginning?",
    },
    {
        "needle_id": "silver_falcon",
        "answer": "SILVER FALCON",
        "fact": "The internal launch password is SILVER FALCON.",
        "question": "What is the internal launch password mentioned at the beginning?",
    },
    {
        "needle_id": "red_lantern",
        "answer": "RED LANTERN",
        "fact": "The backup server alias is RED LANTERN.",
        "question": "What is the backup server alias mentioned at the beginning?",
    },
    {
        "needle_id": "black_river",
        "answer": "BLACK RIVER",
        "fact": "The emergency recovery phrase is BLACK RIVER.",
        "question": "What is the emergency recovery phrase mentioned at the beginning?",
    },
    {
        "needle_id": "golden_tiger",
        "answer": "GOLDEN TIGER",
        "fact": "The customer migration label is GOLDEN TIGER.",
        "question": "What is the customer migration label mentioned at the beginning?",
    },
    {
        "needle_id": "alpha_bridge",
        "answer": "ALPHA BRIDGE",
        "fact": "The deployment checkpoint name is ALPHA BRIDGE.",
        "question": "What is the deployment checkpoint name mentioned at the beginning?",
    },
    {
        "needle_id": "orange_harbor",
        "answer": "ORANGE HARBOR",
        "fact": "The incident tracking marker is ORANGE HARBOR.",
        "question": "What is the incident tracking marker mentioned at the beginning?",
    },
    {
        "needle_id": "purple_anchor",
        "answer": "PURPLE ANCHOR",
        "fact": "The archive restore key is PURPLE ANCHOR.",
        "question": "What is the archive restore key mentioned at the beginning?",
    },
    {
        "needle_id": "white_canyon",
        "answer": "WHITE CANYON",
        "fact": "The database migration token is WHITE CANYON.",
        "question": "What is the database migration token mentioned at the beginning?",
    },
]


def build_needle_prompt(needle: dict) -> str:
    """Build a needle-in-a-haystack prompt for a single candidate fact."""
    needle_text = f"IMPORTANT FACT: {needle['fact']} Remember this value because it will be asked later."

    filler = dedent("""
        This paragraph is unrelated filler text about software engineering,
        memory management, operating systems, compilers, databases, networking,
        and performance optimization. It mentions Rust, Python, Linux, GPUs,
        caches, filesystems, and distributed systems, but it does not contain
        the important value.

        Another unrelated paragraph describes how developers build services,
        debug production issues, write benchmarks, profile latency, optimize
        memory usage, and reason about trade-offs between throughput and quality.
        This paragraph is intentionally noisy and should distract attention from
        the important fact at the beginning.
        """).strip()

    return "\n\n".join([needle_text, filler, f"Question: {needle['question']}\nAnswer:"])


def generate_answer(model, tokenizer, prompt: str) -> str:
    """Greedily decode MAX_NEW_TOKENS tokens for a single prompt, no KV paging."""
    device = next(model.parameters()).device

    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    generated_ids = []

    with torch.inference_mode():
        for _ in range(MAX_NEW_TOKENS):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            generated_ids.append(int(next_token.item()))

            input_ids = torch.cat([input_ids, next_token], dim=-1)

            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device)],
                dim=-1,
            )

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    """Run every needle candidate through the model and report which ones the model recalls."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print("model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager", torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)

    model.eval()

    good = []

    for needle in CANDIDATES:
        prompt = build_needle_prompt(needle)

        tokenized = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

        generated = generate_answer(model, tokenizer, prompt)

        ok = needle["answer"].lower() in generated.lower()

        print(
            "{needle_id:15s} ok={ok:<5} seq_len={seq_len:<4} expected={expected!r} generated={generated!r}".format(
                needle_id=needle["needle_id"],
                ok=str(ok),
                seq_len=tokenized["input_ids"].shape[-1],
                expected=needle["answer"],
                generated=generated,
            )
        )

        if ok:
            good.append(needle)

    print("\nGood needles:")
    for needle in good:
        print(f"- {needle['needle_id']}: {needle['answer']}")


if __name__ == "__main__":
    main()
