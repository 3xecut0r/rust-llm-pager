from __future__ import annotations

import time

import torch
from config import RAM_BUDGET, VRAM_BUDGET
from transformers import AutoModelForCausalLM

from pager_hf import PagedModel
from pager_hf.batched_decode import batched_decode_step

# Core-mechanism proof for Stage 2 serving: batching N independent sessions'
# decode steps into ONE model.forward() call, instead of Stage 1's
# round-robin (one separate call per session per round). Two things to
# prove, on a real model:
# 1. Correctness: each session's generation, advanced via batched_decode_step
#    together with others of *different* real history lengths (exercising
#    the cross-session block-count padding for real, not a trivial
#    equal-length case), must match that same prompt run alone through an
#    ordinary PagedModel.generate() call -- the project's standing bar.
# 2. Real throughput: N sessions advanced via batched_decode_step (one
#    combined forward per round) vs the same N sessions advanced via Stage
#    1's round-robin (one separate forward per session per round) -- this
#    number decides whether Stage 2 is worth wiring into pager_hf.serving at
#    all, not assumed in advance.

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENS_PER_BLOCK = 16
NEW_TOKENS = 6
POLICY = "sinks_heavy_hitter"  # exercises the two-pass mass-tracking kernel path too, not just the simple one

# Priming lengths chosen so every session starts at the SAME tail length (14)
# but a DIFFERENT number of full blocks (0, 1, 2) -- the actual point of this
# mechanism. All three then stay tail-synced every subsequent step too, since
# they all started synced and every step advances every session's tail by
# exactly one token in lockstep.
PROMPT_LENS = [15, 31, 47]


def run_isolated(model, input_ids, attention_mask) -> list[int]:
    session = PagedModel(
        model, vram_budget=VRAM_BUDGET, ram_budget=RAM_BUDGET, policy=POLICY, tokens_per_block=TOKENS_PER_BLOCK
    )
    return session.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=NEW_TOKENS)


def run_batched(model, prompts: list[tuple[torch.Tensor, torch.Tensor]]) -> list[list[int]]:
    sessions = []
    next_tokens = []
    for input_ids, attention_mask in prompts:
        session, _, _ = make_session_from(model, input_ids, attention_mask)
        sessions.append(session)
        next_tokens.append(int(input_ids[0, -1].item()))

    generated = [[] for _ in sessions]
    for _ in range(NEW_TOKENS):
        logits = batched_decode_step(sessions, next_tokens, model.device)
        next_tokens = [int(torch.argmax(row).item()) for row in logits]
        for i, token in enumerate(next_tokens):
            generated[i].append(token)

    return generated


def make_session_from(model, input_ids, attention_mask) -> tuple[PagedModel, torch.Tensor, torch.Tensor]:
    session = PagedModel(
        model, vram_budget=VRAM_BUDGET, ram_budget=RAM_BUDGET, policy=POLICY, tokens_per_block=TOKENS_PER_BLOCK
    )
    session.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=0)
    return session, input_ids, attention_mask


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")
    device = torch.device("cuda")

    print("device:", device)
    print("model:", MODEL_NAME)
    print("prompt_lens:", PROMPT_LENS)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager", torch_dtype=torch.float16).to(
        device
    )
    model.eval()

    prompts = []
    for prompt_len in PROMPT_LENS:
        torch.manual_seed(prompt_len)
        input_ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        prompts.append((input_ids, attention_mask))

    print("\n--- Correctness: batched_decode_step vs isolated PagedModel.generate() ---")
    isolated_ids = [run_isolated(model, ids, mask) for ids, mask in prompts]
    for i, ids in enumerate(isolated_ids):
        print(f"isolated[{i}] (prompt_len={PROMPT_LENS[i]}):", ids)

    batched_ids = run_batched(model, prompts)
    all_ok = True
    for i, (isolated, batched) in enumerate(zip(isolated_ids, batched_ids)):
        matches = isolated == batched
        print(f"session[{i}] matches_isolated={matches} batched={batched}")
        all_ok = all_ok and matches
    assert all_ok, "at least one session's batched generation diverged from its isolated run"
    print("OK: every session matches its isolated PagedModel.generate() run exactly.")

    print("\n--- Throughput: batched_decode_step vs Stage 1 round-robin ---")
    batched_sessions = [make_session_from(model, ids, mask)[0] for ids, mask in prompts]
    batched_next = [int(ids[0, -1].item()) for ids, _ in prompts]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(NEW_TOKENS):
        logits = batched_decode_step(batched_sessions, batched_next, device)
        batched_next = [int(torch.argmax(row).item()) for row in logits]
    torch.cuda.synchronize(device)
    batched_s = time.perf_counter() - started

    round_robin_sessions = [make_session_from(model, ids, mask)[0] for ids, mask in prompts]
    round_robin_next_ids = [ids for ids, _ in prompts]
    round_robin_masks = [mask for _, mask in prompts]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(NEW_TOKENS):
        for i, session in enumerate(round_robin_sessions):
            new_token = session.generate(
                input_ids=round_robin_next_ids[i], attention_mask=round_robin_masks[i], max_new_tokens=1
            )
            round_robin_next_ids[i] = torch.cat(
                [
                    round_robin_next_ids[i],
                    torch.tensor([[new_token[-1]]], dtype=round_robin_next_ids[i].dtype, device=device),
                ],
                dim=1,
            )
            round_robin_masks[i] = torch.cat(
                [round_robin_masks[i], torch.ones((1, 1), dtype=round_robin_masks[i].dtype, device=device)], dim=1
            )
    torch.cuda.synchronize(device)
    round_robin_s = time.perf_counter() - started

    print(f"batched_decode_step total_s ({NEW_TOKENS} steps, {len(prompts)} sessions): {batched_s:.3f}")
    print(f"round_robin total_s        ({NEW_TOKENS} steps, {len(prompts)} sessions): {round_robin_s:.3f}")
    print(f"speedup (round_robin_s / batched_s): {round_robin_s / batched_s:.2f}x")


if __name__ == "__main__":
    main()
