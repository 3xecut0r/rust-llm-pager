from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM

from pager_hf import PagedModel
from pager_hf.batched_decode import batched_decode_step

# Real-scale follow-up to bench/real_kv_batched_decode_mvp.py's toy 3-session,
# 0.5B, dev-GPU proof: same mechanism, a real 7B model, a rented A40, and
# noticeably more concurrent sessions (8, not 3) spread across a wide range
# of real history lengths (0 to 7 full blocks) -- the actual point of
# cross-session padding, exercised at a scale that matters.

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
TOKENS_PER_BLOCK = 16
NEW_TOKENS = 6
POLICY = "sinks_heavy_hitter"
NUM_SESSIONS = 8
VRAM_BUDGET_PER_SESSION = 200_000_000
RAM_BUDGET_PER_SESSION = 500_000_000

# All share tail length 10 (priming_len % 16 == 10) but span 0..7 full blocks --
# priming_len = prompt_len - 1 = 10, 26, 42, ..., 10 + 16*7.
PROMPT_LENS = [10 + 16 * k + 1 for k in range(NUM_SESSIONS)]


def run_isolated(model, input_ids, attention_mask) -> list[int]:
    session = PagedModel(
        model,
        vram_budget=VRAM_BUDGET_PER_SESSION,
        ram_budget=RAM_BUDGET_PER_SESSION,
        policy=POLICY,
        tokens_per_block=TOKENS_PER_BLOCK,
    )
    return session.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=NEW_TOKENS)


def make_session(model, input_ids, attention_mask) -> PagedModel:
    session = PagedModel(
        model,
        vram_budget=VRAM_BUDGET_PER_SESSION,
        ram_budget=RAM_BUDGET_PER_SESSION,
        policy=POLICY,
        tokens_per_block=TOKENS_PER_BLOCK,
    )
    session.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=0)
    return session


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")
    device = torch.device("cuda")

    print("device:", torch.cuda.get_device_name(device))
    print("model:", MODEL_NAME)
    print("num_sessions:", NUM_SESSIONS)
    print("prompt_lens (num_blocks 0..7, shared tail=10):", PROMPT_LENS)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, attn_implementation="eager", torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    prompts = []
    for prompt_len in PROMPT_LENS:
        torch.manual_seed(prompt_len)
        input_ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        prompts.append((input_ids, attention_mask))

    print(f"\n--- Correctness: {NUM_SESSIONS} sessions, batched_decode_step vs isolated PagedModel.generate() ---")
    isolated_ids = [run_isolated(model, ids, mask) for ids, mask in prompts]

    sessions = [make_session(model, ids, mask) for ids, mask in prompts]
    next_tokens = [int(ids[0, -1].item()) for ids, _ in prompts]
    batched_ids = [[] for _ in sessions]

    # Also capture the raw logits (not just the argmax'd token id) for the
    # first step where each session's two paths disagree, so a divergence
    # can be told apart from a genuine bug (large logit gap) vs float
    # noise from a different reduction order in the batched matmul (tiny
    # gap at a near-tie) -- for the FIRST diverging step, not every step,
    # to keep this cheap on real (paid) hardware.
    first_divergence_step = [None] * NUM_SESSIONS
    logit_gap_at_divergence = [None] * NUM_SESSIONS

    for step in range(NEW_TOKENS):
        logits = batched_decode_step(sessions, next_tokens, device)
        next_tokens = [int(torch.argmax(row).item()) for row in logits]
        for i, token in enumerate(next_tokens):
            batched_ids[i].append(token)
            if first_divergence_step[i] is None and token != isolated_ids[i][step]:
                first_divergence_step[i] = step
                sorted_logits = torch.sort(logits[i].float(), descending=True).values
                logit_gap_at_divergence[i] = (sorted_logits[0] - sorted_logits[1]).item()

    # A divergence is only acceptable if it's a genuine near-tie: batched vs
    # isolated matmuls can round differently in bf16 (different reduction
    # order for the same math, a well-known property of batched GPU matmul,
    # not specific to this codebase) -- with a top1-vs-top2 logit gap this
    # small, either token was a coin flip. A real bug would show up as a
    # large, decisive gap picking the "wrong" token anyway.
    NEAR_TIE_THRESHOLD = 0.05
    all_ok = True
    for i, (isolated, batched) in enumerate(zip(isolated_ids, batched_ids)):
        matches = isolated == batched
        detail = ""
        acceptable = matches
        if not matches:
            gap = logit_gap_at_divergence[i]
            acceptable = gap < NEAR_TIE_THRESHOLD
            detail = (
                f" first_diverges_at_step={first_divergence_step[i]} top1_vs_top2_logit_gap={gap:.4f} "
                f"({'near-tie, acceptable' if acceptable else 'NOT a near-tie -- real divergence'}) "
                f"isolated={isolated} batched={batched}"
            )
        print(f"session[{i}] (num_blocks={i}) matches_isolated={matches}{detail}")
        all_ok = all_ok and acceptable
    if not all_ok:
        print("FAIL: at least one session diverged with a large logit gap -- a real bug, not float noise.")
    else:
        print(f"OK: all {NUM_SESSIONS} sessions match exactly, or diverge only at a genuine bf16 near-tie.")
    assert all_ok, "at least one session diverged with a large logit gap -- see the FAIL line above"

    print(f"\n--- Throughput: {NUM_SESSIONS} sessions, batched_decode_step vs round-robin ---")
    batched_sessions = [make_session(model, ids, mask) for ids, mask in prompts]
    batched_next = [int(ids[0, -1].item()) for ids, _ in prompts]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(NEW_TOKENS):
        logits = batched_decode_step(batched_sessions, batched_next, device)
        batched_next = [int(torch.argmax(row).item()) for row in logits]
    torch.cuda.synchronize(device)
    batched_s = time.perf_counter() - started

    round_robin_sessions = [make_session(model, ids, mask) for ids, mask in prompts]
    round_robin_ids = [ids.clone() for ids, _ in prompts]
    round_robin_masks = [mask.clone() for _, mask in prompts]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(NEW_TOKENS):
        for i, session in enumerate(round_robin_sessions):
            new_token = session.generate(
                input_ids=round_robin_ids[i], attention_mask=round_robin_masks[i], max_new_tokens=1
            )
            round_robin_ids[i] = torch.cat(
                [round_robin_ids[i], torch.tensor([[new_token[-1]]], dtype=round_robin_ids[i].dtype, device=device)],
                dim=1,
            )
            round_robin_masks[i] = torch.cat(
                [round_robin_masks[i], torch.ones((1, 1), dtype=round_robin_masks[i].dtype, device=device)], dim=1
            )
    torch.cuda.synchronize(device)
    round_robin_s = time.perf_counter() - started

    print(f"batched_decode_step total_s ({NEW_TOKENS} steps, {NUM_SESSIONS} sessions): {batched_s:.3f}")
    print(f"round_robin total_s        ({NEW_TOKENS} steps, {NUM_SESSIONS} sessions): {round_robin_s:.3f}")
    print(f"speedup (round_robin_s / batched_s): {round_robin_s / batched_s:.2f}x")


if __name__ == "__main__":
    main()
