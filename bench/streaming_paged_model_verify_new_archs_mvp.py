from __future__ import annotations

import torch
from config import PROMOTE_MARGIN, RAM_BUDGET, RAM_PROMOTE_MARGIN, REBALANCE_INTERVAL, RECENT_WINDOW, VRAM_BUDGET
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache

from pager_hf import PagedModel

# Same verification as streaming_paged_model_verify_llama_mvp.py /
# _mistral_mvp.py, for the architectures added on top of Qwen2/Llama/Mistral:
# Qwen2MoE and Starcoder2 (qwen2-style RoPE, self_attn is a straight copy of
# Qwen2Attention -- MoE only changes the MLP; their sliding-window config
# defaults off, and this transformers version's eager-mode causal-mask builder
# doesn't enforce one anyway, so there's nothing to replicate), Gemma
# (llama-style RoPE, no gotchas in this transformers version), Phi3 (fused
# qkv_proj, otherwise llama-style RoPE -- see paged_model._project_qkv), and
# Gemma2 (non-default QK^T scale + attn-logit softcapping, both baked into the
# Triton kernels -- see paged_model._scale_and_softcap -- plus real
# sliding-window attention on alternating layers; PROMPT_LEN/GENERATE_TOKENS
# here stay far under any real sliding_window value, so this specific test
# doesn't exercise the window truncation itself -- that's covered separately,
# with a deliberately tiny window, in
# streaming_paged_model_verify_sliding_window_mvp.py). All tiny checkpoints
# have random (untrained) weights, so the generated text is meaningless -- the
# bar here is mechanical: does streaming attention compute the exact same
# thing as baseline through the real attention code path, not whether the
# output reads well.
TOKENS_PER_BLOCK = 16
PROMPT_LEN = 80
GENERATE_TOKENS = 12
POLICIES = ["recent_only", "sinks_recent", "heavy_hitter", "sinks_heavy_hitter"]

MODELS = [
    "katuni4ka/tiny-random-qwen1.5-moe",  # model_type qwen2_moe -- "Qwen1.5-MoE" was the release name
    "optimum-intel-internal-testing/tiny-random-Starcoder2ForCausalLM",
    "Xenova/tiny-random-GemmaForCausalLM",
    "Xenova/tiny-random-Phi3ForCausalLM",
    "katuni4ka/tiny-random-gemma2",
]


def greedy_baseline_generate(*, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int) -> list[int]:
    """
    Explicit position_ids/cache_position at every call, not HF's own defaults
    (torch.arange(0, chunk_len), i.e. always restarting from 0 regardless of
    how much is already cached) -- most architectures' causal-mask building
    doesn't depend on cache_position's absolute value for a plain
    DynamicCache, so this omission was invisible until Gemma2 (whose
    _update_causal_mask uses cache_position directly) exposed it: an unset
    cache_position built the wrong mask and silently produced a
    wrong-but-plausible "baseline", not a real reference at all. Also passes
    an explicit empty DynamicCache rather than leaving past_key_values unset --
    Gemma2Model.forward (confirmed by reading its source) never auto-wraps a
    None past_key_values into one itself, unlike every other supported
    architecture, so an omitted cache here silently never caches anything at
    all (same root cause as PagedModel._chunked_prefill's own fix).
    """
    prefix_len = input_ids.shape[-1] - 1
    device = input_ids.device

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids[:, :prefix_len],
            attention_mask=attention_mask[:, :prefix_len],
            position_ids=torch.arange(prefix_len, device=device).unsqueeze(0),
            cache_position=torch.arange(prefix_len, device=device),
            past_key_values=DynamicCache(),
            use_cache=True,
            output_attentions=False,
        )

    current_cache = outputs.past_key_values
    next_input_id = input_ids[:, prefix_len:]
    current_len = prefix_len
    current_attention_mask = attention_mask[:, : prefix_len + 1]
    generated: list[int] = []

    with torch.inference_mode():
        for _ in range(steps):
            outputs = model(
                input_ids=next_input_id,
                attention_mask=current_attention_mask,
                position_ids=torch.tensor([[current_len]], device=device),
                cache_position=torch.tensor([current_len], device=device),
                past_key_values=current_cache,
                use_cache=True,
                output_attentions=False,
            )
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            generated.append(int(next_token_id.item()))

            current_cache = outputs.past_key_values
            next_input_id = next_token_id
            current_len += 1
            current_attention_mask = torch.cat(
                [
                    current_attention_mask,
                    torch.ones((current_attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device),
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


def verify_model(model_name: str) -> bool:
    device = torch.device("cuda")
    print("\n" + "=" * 60)
    print("model:", model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="eager", torch_dtype=torch.float16).to(
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

    # These tiny checkpoints have random (untrained) weights and, for at
    # least one of them, a tokenizer.json in a format this project's pinned
    # `tokenizers` version can't parse -- neither matters for a purely
    # mechanical check (does streaming attention compute the exact same
    # thing as baseline), so random token ids straight from the model's own
    # vocab range replace the usual real-tokenizer prompt.
    torch.manual_seed(0)
    input_ids = torch.randint(0, model.config.vocab_size, (1, PROMPT_LEN), device=device)
    attention_mask = torch.ones_like(input_ids)
    print("prompt_seq_len:", input_ids.shape[-1])

    baseline_generated = greedy_baseline_generate(
        model=model, input_ids=input_ids, attention_mask=attention_mask, steps=GENERATE_TOKENS
    )
    print("baseline_ids:", baseline_generated)

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

        print(f"policy={policy}: matches_baseline={matches_baseline} matches_non_streaming={matches_non_streaming}")

    return all_ok


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    results = {model_name: verify_model(model_name) for model_name in MODELS}

    print("\n" + "=" * 60)
    print("Summary")
    print("-------")
    for model_name, ok in results.items():
        print(f"{model_name}: {'OK' if ok else 'FAIL'}")

    if not all(results.values()):
        raise SystemExit(1)
    print(
        "\nOK: streaming PagedModel matches baseline and non-streaming PagedModel for all 4 policies, "
        "on Qwen2MoE, Starcoder2, Gemma, Phi3, and Gemma2."
    )


if __name__ == "__main__":
    main()
