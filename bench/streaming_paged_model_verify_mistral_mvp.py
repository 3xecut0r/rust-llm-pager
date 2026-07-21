from __future__ import annotations

import torch
from config import PROMOTE_MARGIN, RAM_BUDGET, RAM_PROMOTE_MARGIN, REBALANCE_INTERVAL, RECENT_WINDOW, VRAM_BUDGET
from real_kv_utils import build_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer

from pager_hf import PagedModel

# Same verification as streaming_paged_model_verify_llama_mvp.py, but for Mistral --
# proves _compute_rope's "llama-style" branch (paged_model.py) also covers Mistral's
# RoPE convention, not just Llama's. hf-internal-testing/tiny-random-MistralForCausalLM
# has random (untrained) weights, so the generated text is meaningless -- that's fine,
# the bar here is mechanical: does streaming attention compute the exact same thing as
# baseline through the real MistralAttention code path, not whether the output reads well.
MODEL_NAME = "hf-internal-testing/tiny-random-MistralForCausalLM"
TOKENS_PER_BLOCK = 16
MAX_LENGTH = 96  # config.max_position_embeddings is only 512 for this tiny checkpoint
GENERATE_TOKENS = 12
POLICIES = ["recent_only", "sinks_recent", "heavy_hitter", "sinks_heavy_hitter"]


def greedy_baseline_generate(*, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int) -> list[int]:
    next_input_id = input_ids[:, -1:]
    current_attention_mask = attention_mask

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids[:, :-1], attention_mask=attention_mask[:, :-1], use_cache=True, output_attentions=False
        )

    current_cache = outputs.past_key_values
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_attention_mask,
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(next_token_id.item()))

            current_cache = outputs.past_key_values
            next_input_id = next_token_id
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones(
                        (current_attention_mask.shape[0], 1),
                        dtype=current_attention_mask.dtype,
                        device=current_attention_mask.device,
                    ),
                ],
                dim=1,
            )

    return generated


def run_policy(*, model, policy: str, use_streaming: bool, input_ids, attention_mask, steps: int) -> list[int]:
    paged_model = PagedModel(
        model,
        vram_budget=VRAM_BUDGET,
        ram_budget=RAM_BUDGET,
        recent_window=RECENT_WINDOW,
        rebalance_interval=REBALANCE_INTERVAL,
        promote_margin=PROMOTE_MARGIN,
        ram_promote_margin=RAM_PROMOTE_MARGIN,
        policy=policy,
        tokens_per_block=TOKENS_PER_BLOCK,
        use_streaming_attention=use_streaming,
        streaming_group_size_blocks=2,
    )
    return paged_model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=steps)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    device = torch.device("cuda")
    print("device:", device)
    print("model:", MODEL_NAME)
    print("generate_tokens:", GENERATE_TOKENS)
    print("tokens_per_block:", TOKENS_PER_BLOCK)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="eager", torch_dtype=torch.float16).to(
        device
    )
    model.eval()
    print("model_type:", model.config.model_type)
    print(
        "num_hidden_layers:",
        model.config.num_hidden_layers,
        "num_attention_heads:",
        model.config.num_attention_heads,
        "num_key_value_heads:",
        model.config.num_key_value_heads,
    )

    prompt = build_prompt()
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    print("prompt_seq_len:", input_ids.shape[-1])

    baseline_generated = greedy_baseline_generate(
        model=model, input_ids=input_ids, attention_mask=attention_mask, steps=GENERATE_TOKENS
    )
    print("\nbaseline_ids:", baseline_generated)

    all_ok = True
    for policy in POLICIES:
        non_streaming_ids = run_policy(
            model=model,
            policy=policy,
            use_streaming=False,
            input_ids=input_ids,
            attention_mask=attention_mask,
            steps=GENERATE_TOKENS,
        )
        streaming_ids = run_policy(
            model=model,
            policy=policy,
            use_streaming=True,
            input_ids=input_ids,
            attention_mask=attention_mask,
            steps=GENERATE_TOKENS,
        )

        matches_baseline = streaming_ids == baseline_generated
        matches_non_streaming = streaming_ids == non_streaming_ids
        all_ok = all_ok and matches_baseline and matches_non_streaming

        print(f"\npolicy: {policy}")
        print("non_streaming_ids:", non_streaming_ids)
        print("streaming_ids:    ", streaming_ids)
        print("streaming_matches_baseline:", matches_baseline)
        print("streaming_matches_non_streaming_paged:", matches_non_streaming)

    if all_ok:
        print(
            "\nOK: streaming PagedModel matches baseline and non-streaming PagedModel for all 4 policies on Mistral too."
        )
    else:
        print("\nFAIL: at least one policy diverged.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
