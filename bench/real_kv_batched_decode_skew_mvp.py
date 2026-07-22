from __future__ import annotations

import time

import torch
from config import RAM_BUDGET, VRAM_BUDGET
from transformers import AutoModelForCausalLM

from pager_hf import PagedModel
from pager_hf.batched_decode import batched_decode_step

# Point-proving benchmark for the per-row padding-skip optimization in
# batched_decode.py/streaming_attention.py: does adding one SHORT session to
# a batch of LONG ones cost close to nothing (the goal), or close to what
# another LONG session would cost (the bug this fixes)?
#
# Method: time a batch of K long sessions alone, then time the SAME K long
# sessions plus one extra short session in the same round (same max_blocks,
# since the long sessions are unchanged) -- the short session's marginal
# cost is what this optimization targets. Compare that marginal cost against
# the average per-session cost within the long-only batch: if the skip
# works, the short session should cost much less than an "average" row,
# even though every row's gathered K/V tensor is still padded to max_blocks.

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TOKENS_PER_BLOCK = 16
NEW_TOKENS = 6
POLICY = "sinks_heavy_hitter"
NUM_LONG_SESSIONS = 4
LONG_NUM_BLOCKS = 24
SHORT_NUM_BLOCKS = 1
TAIL_REMAINDER = 10  # shared priming_len % tokens_per_block for every session


def prompt_len_for(num_blocks: int) -> int:
    return TAIL_REMAINDER + TOKENS_PER_BLOCK * num_blocks + 1


def make_session(model, input_ids, attention_mask) -> PagedModel:
    session = PagedModel(
        model, vram_budget=VRAM_BUDGET, ram_budget=RAM_BUDGET, policy=POLICY, tokens_per_block=TOKENS_PER_BLOCK
    )
    session.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=0)
    return session


def build_prompts(model, device, num_blocks_list: list[int]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    prompts = []
    for num_blocks in num_blocks_list:
        prompt_len = prompt_len_for(num_blocks)
        torch.manual_seed(prompt_len)
        input_ids = torch.randint(0, model.config.vocab_size, (1, prompt_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        prompts.append((input_ids, attention_mask))
    return prompts


def time_batched_rounds(model, device, prompts: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    sessions = [make_session(model, ids, mask) for ids, mask in prompts]
    next_tokens = [int(ids[0, -1].item()) for ids, _ in prompts]
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for _ in range(NEW_TOKENS):
        logits = batched_decode_step(sessions, next_tokens, device)
        next_tokens = [int(torch.argmax(row).item()) for row in logits]
    torch.cuda.synchronize(device)
    return time.perf_counter() - started


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")
    device = torch.device("cuda")

    print("device:", torch.cuda.get_device_name(device))
    print("model:", MODEL_NAME)
    print(f"long sessions: {NUM_LONG_SESSIONS} x {LONG_NUM_BLOCKS} blocks, short session: {SHORT_NUM_BLOCKS} block")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager", torch_dtype=torch.float16).to(
        device
    )
    model.eval()

    long_only_prompts = build_prompts(model, device, [LONG_NUM_BLOCKS] * NUM_LONG_SESSIONS)
    skewed_prompts = long_only_prompts + build_prompts(model, device, [SHORT_NUM_BLOCKS])

    # Warm up Triton's autotuner/compilation cache before timing either round,
    # so the first measured round isn't paying a one-time compile cost.
    time_batched_rounds(model, device, long_only_prompts)

    long_only_s = time_batched_rounds(model, device, long_only_prompts)
    skewed_s = time_batched_rounds(model, device, skewed_prompts)

    avg_long_session_s = long_only_s / NUM_LONG_SESSIONS
    marginal_short_session_s = skewed_s - long_only_s

    print(f"\nlong_only_s ({NUM_LONG_SESSIONS} sessions, {NEW_TOKENS} steps): {long_only_s:.3f}")
    print(f"skewed_s    ({NUM_LONG_SESSIONS}+1 sessions, {NEW_TOKENS} steps): {skewed_s:.3f}")
    print(f"avg cost of one long session in the long-only round: {avg_long_session_s:.4f}s")
    print(f"marginal cost of adding the short session: {marginal_short_session_s:.4f}s")
    print(
        f"marginal short-session cost as a fraction of an average long session's cost: "
        f"{marginal_short_session_s / avg_long_session_s:.2f}x"
    )
    print(
        "\nIf the padding-skip optimization works, the short session's marginal cost should be "
        "well under an average long session's cost, despite every row still being gathered into a "
        f"{LONG_NUM_BLOCKS}-block-wide tensor -- the short session's kernel program skips almost all of it."
    )


if __name__ == "__main__":
    main()
