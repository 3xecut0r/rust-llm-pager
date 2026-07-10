from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from textwrap import dedent

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import pager


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

TOKENS_PER_BLOCK = 1

VRAM_BUDGET = 256_000_000
RAM_BUDGET = 512_000_000

RECENT_WINDOW = 64
REBALANCE_INTERVAL = 32
PROMOTE_MARGIN = 0.05
RAM_PROMOTE_MARGIN = 0.20

MAX_LENGTH = 256

NEEDLES = [
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
]

RAM_ATTENTION_PENALTIES = [-0.5, -1.0, -2.0, -4.0]
SSD_ATTENTION_PENALTIES = [-2.0, -4.0, -6.0, -8.0]

RAM_ATTENTION_PENALTY = -2.0
SSD_ATTENTION_PENALTY = -6.0

POLICIES = [
    "full_context",
    "recent_only",
    "sinks_recent",
    "heavy_hitter",
    "sinks_heavy_hitter",
]

OUT_PATH = Path("bench/multi_needle_soft_masked_grid_results.csv")


def metrics_to_dict(m):
    ratio = (
        m.attention_mass_vram / m.attention_mass_total
        if m.attention_mass_total
        else 0
    )

    return {
        "tokens": m.tokens,
        "vram_peak": m.vram_peak,
        "ram_peak": m.ram_peak,
        "swap_vram_ram": m.swap_vram_ram,
        "swap_ram_ssd": m.swap_ram_ssd,
        "attention_mass_total": m.attention_mass_total,
        "attention_mass_vram": m.attention_mass_vram,
        "vram_attention_ratio": ratio,
    }


def token_attention_to_blocks(
        attn_to_keys: torch.Tensor,
        tokens_per_block: int,
) -> list[float]:
    seq_len = attn_to_keys.shape[0]
    blocks = (seq_len + tokens_per_block - 1) // tokens_per_block

    out = [0.0 for _ in range(blocks)]

    for token_idx in range(seq_len):
        block_idx = token_idx // tokens_per_block
        out[block_idx] += float(attn_to_keys[token_idx].item())

    return out


def build_needle_prompt(needle: dict) -> str:
    fact = needle["fact"]

    needle_text = (
        f"IMPORTANT FACT: {fact} "
        "Remember this value because it will be asked later."
    )

    filler_unit = dedent(
        """
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
        """
    ).strip()

    filler = "\n\n".join([filler_unit for _ in range(1)])

    question = (
        f"Question: {needle['question']}\n"
        "Answer:"
    )

    return "\n\n".join([needle_text, filler, question])


def extract_last_query_traces(outputs) -> list[tuple[int, list[float]]]:
    attentions = outputs.attentions

    if attentions is None:
        raise RuntimeError("Model did not return attentions.")

    traces = []

    for layer_idx, layer_attn in enumerate(attentions):
        layer_attn = layer_attn.detach().float().cpu()

        q_attn = layer_attn[0, :, -1, :]
        attn_to_keys = q_attn.mean(dim=0)

        block_attn = token_attention_to_blocks(
            attn_to_keys,
            TOKENS_PER_BLOCK,
        )

        traces.append((layer_idx, block_attn))

    return traces


def build_tier_attention_bias(
        *,
        base_attention_mask: torch.Tensor,
        tiers: list[int],
) -> torch.Tensor:
    """
    Build a 4D additive attention bias.

    Shape:
        [batch, 1, query_len, key_len]

    Values:
        0.0   for VRAM tokens
        -2.0  for RAM tokens
        -6.0  for SSD tokens
        -1e4  for padding/future-masked tokens

    This is a soft simulator:
    RAM/SSD tokens are not fully hidden, but their attention logits are penalized.
    """
    batch_size, seq_len = base_attention_mask.shape
    device = base_attention_mask.device
    dtype = torch.float16 if base_attention_mask.device.type == "cuda" else torch.float32

    bias = torch.zeros(
        (batch_size, 1, seq_len, seq_len),
        device=device,
        dtype=dtype,
    )

    key_penalties = torch.zeros(
        (seq_len,),
        device=device,
        dtype=dtype,
    )

    for token_idx in range(seq_len):
        block_idx = token_idx // TOKENS_PER_BLOCK

        if block_idx >= len(tiers):
            key_penalties[token_idx] = SSD_ATTENTION_PENALTY
            continue

        tier = tiers[block_idx]

        if tier == 0:
            key_penalties[token_idx] = 0.0
        elif tier == 1:
            key_penalties[token_idx] = RAM_ATTENTION_PENALTY
        else:
            key_penalties[token_idx] = SSD_ATTENTION_PENALTY

    # Apply per-key tier penalty to every query row.
    bias = bias + key_penalties.view(1, 1, 1, seq_len)

    # Causal mask: future keys invisible.
    causal = torch.triu(
        torch.ones((seq_len, seq_len), device=device, dtype=torch.bool),
        diagonal=1,
    )
    bias[:, :, causal] = -1e4

    # Padding mask.
    padding_mask = base_attention_mask == 0
    if padding_mask.any():
        bias = bias.masked_fill(
            padding_mask.view(batch_size, 1, 1, seq_len),
            -1e4,
        )

    # Safety anchors.
    # Token 0 and current token should not be hard-masked.
    bias[:, :, :, 0] = torch.maximum(
        bias[:, :, :, 0],
        torch.tensor(0.0, device=device, dtype=dtype),
    )
    bias[:, :, -1, -1] = 0.0

    return bias


def score_expected_answer(
        *,
        model,
        tokenizer,
        prompt: str,
        policy: str,
        expected_answer: str,
        needle_id: str,
) -> dict:
    device = next(model.parameters()).device

    if device.type == "cuda":
        torch.cuda.empty_cache()

    prompt_encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    expected_answer_with_space = " " + expected_answer

    answer_encoded = tokenizer(
        expected_answer_with_space,
        return_tensors="pt",
        add_special_tokens=False,
    )

    input_ids = prompt_encoded["input_ids"].to(device)
    attention_mask = prompt_encoded["attention_mask"].to(device)
    answer_ids = answer_encoded["input_ids"][0].to(device)

    p = None
    if policy != "full_context":
        p = pager.PyPager(
            VRAM_BUDGET,
            RAM_BUDGET,
            RECENT_WINDOW,
            REBALANCE_INTERVAL,
            PROMOTE_MARGIN,
            RAM_PROMOTE_MARGIN,
            policy,
        )

    logprobs = []
    predicted_ids = []

    started = time.perf_counter()

    with torch.inference_mode():
        for target_id in answer_ids:
            current_query_idx = int(input_ids.shape[-1] - 1)

            if policy == "full_context":
                scoring_attention_mask = attention_mask
            else:
                # Pass 1: full attention trace for policy update.
                trace_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    use_cache=False,
                )

                traces = extract_last_query_traces(trace_outputs)

                for layer_idx, block_attn in traces:
                    p.on_step(current_query_idx, layer_idx, block_attn)

                tiers = p.tiers()

                scoring_attention_mask = build_tier_attention_bias(
                    base_attention_mask=attention_mask,
                    tiers=tiers,
                )

                del trace_outputs

                if device.type == "cuda":
                    torch.cuda.empty_cache()

            # Pass 2: score expected next token under selected mask.
            score_outputs = model(
                input_ids=input_ids,
                attention_mask=scoring_attention_mask,
                output_attentions=False,
                use_cache=False,
            )

            logits = score_outputs.logits[:, -1, :]

            if torch.isnan(logits).any():
                print(f"WARNING: NaN logits for policy={policy}, query_idx={current_query_idx}")

            logits = torch.nan_to_num(
                logits,
                nan=-1e4,
                posinf=1e4,
                neginf=-1e4,
            )

            log_probs = F.log_softmax(logits, dim=-1)

            token_logprob = float(log_probs[0, int(target_id.item())].item())
            logprobs.append(token_logprob)

            predicted_id = int(torch.argmax(logits, dim=-1).item())
            predicted_ids.append(predicted_id)

            # Teacher forcing: append correct token, not predicted token.
            next_token = target_id.view(1, 1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)

            next_mask = torch.ones(
                (attention_mask.shape[0], 1),
                dtype=attention_mask.dtype,
                device=device,
            )
            attention_mask = torch.cat([attention_mask, next_mask], dim=-1)

            del score_outputs, logits, log_probs

            if device.type == "cuda":
                torch.cuda.empty_cache()

    elapsed = time.perf_counter() - started

    total_logprob = sum(logprobs)
    avg_logprob = total_logprob / len(logprobs)
    perplexity = math.exp(-avg_logprob)

    predicted_text = tokenizer.decode(
        predicted_ids,
        skip_special_tokens=True,
    )

    if p is None:
        metrics = {
            "tokens": 0,
            "vram_peak": 0,
            "ram_peak": 0,
            "swap_vram_ram": 0,
            "swap_ram_ssd": 0,
            "attention_mass_total": 0.0,
            "attention_mass_vram": 0.0,
            "vram_attention_ratio": 1.0,
        }
    else:
        metrics = metrics_to_dict(p.metrics())

    swap_total_gb = (
                            metrics["swap_vram_ram"] + metrics["swap_ram_ssd"]
                    ) / 1_000_000_000

    return {
        "needle_id": needle_id,
        "policy": policy,
        "expected_answer": expected_answer_with_space,
        "answer_tokens": len(answer_ids),
        "total_logprob": total_logprob,
        "avg_logprob": avg_logprob,
        "perplexity": perplexity,
        "predicted_text_greedy_per_step": predicted_text,
        "elapsed_sec": elapsed,
        "swap_total_gb": swap_total_gb,
        **metrics,
    }


def main():
    global RAM_ATTENTION_PENALTY, SSD_ATTENTION_PENALTY

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("device:", device)
    print("model:", MODEL_NAME)
    print("eval: soft-masked teacher-forced logprob")
    print("ram_penalties:", RAM_ATTENTION_PENALTIES)
    print("ssd_penalties:", SSD_ATTENTION_PENALTIES)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        attn_implementation="eager",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    model.eval()

    results = []

    for needle in NEEDLES:
        needle_id = needle["needle_id"]
        expected_answer = needle["answer"]

        print("\n" + "=" * 80)
        print(f"Needle: {needle_id}")
        print(f"Expected answer: {expected_answer!r}")

        prompt = build_needle_prompt(needle)

        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        decoded_prompt = tokenizer.decode(
            tokenized["input_ids"][0],
            skip_special_tokens=True,
        )

        print("contains expected answer:", expected_answer in decoded_prompt)
        print("contains Question:", "Question:" in decoded_prompt)
        print("contains Answer:", "Answer:" in decoded_prompt)
        print("prompt seq_len:", tokenized["input_ids"].shape[-1])

        for ram_penalty in RAM_ATTENTION_PENALTIES:
            for ssd_penalty in SSD_ATTENTION_PENALTIES:
                if ssd_penalty > ram_penalty:
                    continue

                RAM_ATTENTION_PENALTY = ram_penalty
                SSD_ATTENTION_PENALTY = ssd_penalty

                print(
                    f"\nPenalty setting: RAM={RAM_ATTENTION_PENALTY}, "
                    f"SSD={SSD_ATTENTION_PENALTY}"
                )

                for policy in POLICIES:
                    print(f"Scoring policy: {policy}")

                    row = score_expected_answer(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        policy=policy,
                        expected_answer=expected_answer,
                        needle_id=needle_id,
                    )

                    row["ram_attention_penalty"] = RAM_ATTENTION_PENALTY
                    row["ssd_attention_penalty"] = SSD_ATTENTION_PENALTY

                    results.append(row)

                    print(
                        "needle={needle} policy={policy} ppl={ppl:.3f} avg_logprob={avg:.3f} "
                        "ratio={ratio:.3f} swap={swap:.3f}GB greedy_steps={greedy!r}".format(
                            needle=needle_id,
                            policy=policy,
                            ppl=row["perplexity"],
                            avg=row["avg_logprob"],
                            ratio=row["vram_attention_ratio"],
                            swap=row["swap_total_gb"],
                            greedy=row["predicted_text_greedy_per_step"],
                        )
                    )

    print("\nSummary:")
    for row in results:
        print(
            "{policy:13s} ppl={ppl:.3f} avg_logprob={avg:.3f} "
            "ratio={ratio:.3f} swap={swap:.3f}GB".format(
                policy=row["policy"],
                ppl=row["perplexity"],
                avg=row["avg_logprob"],
                ratio=row["vram_attention_ratio"],
                swap=row["swap_total_gb"],
            )
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
