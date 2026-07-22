from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, Gemma2Config, MistralConfig
from transformers.cache_utils import DynamicCache

from pager_hf import PagedModel

# Dedicated correctness test for real sliding-window truncation
# (paged_model._sliding_window_for_layer), the part of this pass that none of
# the existing architecture-verification benches actually exercise: every
# prompt elsewhere in this project is far shorter than any real checkpoint's
# sliding_window (typically 4096), so the window bound is a no-op there even
# when the mechanism is wired up correctly. Building a tiny, from-scratch,
# randomly-initialized config with an artificially small sliding_window (16)
# and a prompt that genuinely exceeds it is the only way to actually prove
# the truncation logic works, rather than assuming it from the arithmetic.
#
# Mistral applies its config.sliding_window uniformly to every layer, and its
# real eager-mode HF baseline enforces it via the model-level causal mask
# (confirmed by reading MistralModel._update_causal_mask) -- this was a real,
# previously-silent gap in this project's already-shipped Mistral support.
# Gemma2 runs sliding-window attention on alternating layers (confirmed via
# Gemma2Attention.__init__'s own self.sliding_window / Gemma2DecoderLayer's
# is_sliding-gated mask logic).
TOKENS_PER_BLOCK = 16  # tl.dot's contraction dim (P.V's is tokens_per_block) must be >= 16
SLIDING_WINDOW = 24
PROMPT_LEN = 64  # well past SLIDING_WINDOW, so the window bound genuinely excludes real history
GENERATE_TOKENS = 8
POLICIES = ["recent_only", "sinks_recent", "heavy_hitter", "sinks_heavy_hitter"]

MISTRAL_CONFIG = MistralConfig(
    vocab_size=1000,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    sliding_window=SLIDING_WINDOW,
    max_position_embeddings=256,
)
GEMMA2_CONFIG = Gemma2Config(
    vocab_size=1000,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=4,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    sliding_window=SLIDING_WINDOW,
    query_pre_attn_scalar=16,
    attn_logit_softcapping=50.0,
    final_logit_softcapping=30.0,
    max_position_embeddings=256,
)


def careful_baseline_generate(
    *, model, input_ids: torch.Tensor, attention_mask: torch.Tensor, steps: int
) -> tuple[list[int], list[float]]:
    """Full recompute from scratch at every step (fresh DynamicCache, no cache reuse across
    steps, q_len == kv_len on every call) rather than one-token-at-a-time incremental decode.

    This is deliberately more expensive than the usual incremental-decode baseline used
    elsewhere in this project, and for a reason specific to this test: transformers==4.44.2's
    Gemma2DecoderLayer.forward builds its sliding-window exclusion via
    `torch.tril(torch.ones_like(attention_mask), diagonal=-sliding_window)`, which operates on
    the *local* row index of the query dimension, not the absolute sequence position baked into
    the mask's cell values. During prefill (q_len == kv_len) that local row index happens to
    coincide with the absolute position, so the window is enforced correctly -- but during
    single-token incremental decode (q_len == 1, kv_len == however much is cached), the only row
    index is 0, so `col <= 0 - sliding_window` can never hold for a positive window and the
    tril mask is always all-False: the window is silently never enforced past the first prefill
    chunk. Confirmed directly by hooking Gemma2Attention.forward and inspecting the actual
    attention_mask tensor it receives on a decode step -- no -inf anywhere, even far outside the
    window. Recomputing the whole sequence-so-far every step (always q_len == kv_len) sidesteps
    this HF bug entirely, giving a ground truth where the window is genuinely enforced at every
    step, which is the only way to test this project's own real window-truncation logic against
    something that actually enforces a window. Mistral's masking (built once, model-level, in
    MistralModel._update_causal_mask, using cache_position directly rather than local row index)
    does not have this bug -- this generic implementation is correct for it either way."""
    device = input_ids.device
    current_ids = input_ids
    current_mask = attention_mask
    generated: list[int] = []
    top1_vs_top2_gaps: list[float] = []

    with torch.inference_mode():
        for _ in range(steps):
            seq_len = current_ids.shape[-1]
            outputs = model(
                input_ids=current_ids,
                attention_mask=current_mask,
                position_ids=torch.arange(seq_len, device=device).unsqueeze(0),
                cache_position=torch.arange(seq_len, device=device),
                past_key_values=DynamicCache(),
                use_cache=True,
                output_attentions=False,
            )
            logits = outputs.logits[:, -1, :].float()
            sorted_logits = torch.sort(logits[0], descending=True).values
            top1_vs_top2_gaps.append((sorted_logits[0] - sorted_logits[1]).item())
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
            generated.append(int(next_token_id.item()))

            current_ids = torch.cat([current_ids, next_token_id], dim=1)
            current_mask = torch.cat(
                [current_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)], dim=1
            )

    return generated, top1_vs_top2_gaps


def verify_model(name: str, config) -> bool:
    device = torch.device("cuda")
    print("\n" + "=" * 60)
    print("model:", name, "sliding_window:", SLIDING_WINDOW, "prompt_len:", PROMPT_LEN)

    # Seed before model creation too, not just before building input_ids --
    # AutoModelForCausalLM.from_config()'s random weight init consumes the
    # global RNG state, so without this the model's own weights (not just the
    # prompt) are a different random draw on every run, making any
    # correctness comparison irreproducible.
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, attn_implementation="eager", torch_dtype=torch.float16).to(device)
    model.eval()

    torch.manual_seed(0)
    input_ids = torch.randint(0, config.vocab_size, (1, PROMPT_LEN), device=device)
    attention_mask = torch.ones_like(input_ids)

    baseline, gaps = careful_baseline_generate(
        model=model, input_ids=input_ids, attention_mask=attention_mask, steps=GENERATE_TOKENS
    )
    print("baseline (window genuinely enforced):", baseline)
    print("baseline top1_vs_top2 gaps:", [round(g, 4) for g in gaps])

    # A divergence is only acceptable if the baseline's OWN top1-vs-top2 gap at
    # that exact step was already a near-tie -- the same diagnostic this
    # project has used before (e.g. bench/real_kv_batched_decode_scale_mvp.py):
    # tiny, randomly-initialized checkpoints produce flat, tie-prone logit
    # distributions, and Gemma2's attn/final-logit softcapping (tanh-based,
    # saturating) makes the eventual argmax unusually sensitive to ordinary
    # reduction-order differences between the streaming (Triton tensor-core)
    # and dense (cuBLAS) matmul paths -- both mathematically correct, neither
    # deterministically "more right". A large gap flipping would be a real bug.
    NEAR_TIE_THRESHOLD = 0.01

    all_ok = True
    for policy in POLICIES:
        paged_model = PagedModel(
            model,
            vram_budget=128_000_000,
            ram_budget=2_000_000_000,
            policy=policy,
            tokens_per_block=TOKENS_PER_BLOCK,
            use_streaming_attention=True,
            streaming_group_size_blocks=2,
        )
        streaming_ids = paged_model.generate(
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=GENERATE_TOKENS
        )
        matches = streaming_ids == baseline
        acceptable = matches
        detail = ""
        if not matches:
            first_diff = next(i for i, (a, b) in enumerate(zip(baseline, streaming_ids)) if a != b)
            gap = gaps[first_diff]
            acceptable = gap < NEAR_TIE_THRESHOLD
            detail = f" first_diverges_at_step={first_diff} gap_there={gap:.4f} ({'near-tie, acceptable' if acceptable else 'NOT a near-tie'})"
        print(f"policy={policy}: matches_baseline={matches}{detail} streaming_ids={streaming_ids}")
        all_ok = all_ok and acceptable

    return all_ok


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this MVP.")

    results = {
        "mistral (uniform window, real eager baseline already enforces it)": verify_model(
            "mistral-tiny-random-sliding-window", MISTRAL_CONFIG
        ),
        "gemma2 (alternating window, layer-level is_sliding)": verify_model(
            "gemma2-tiny-random-sliding-window", GEMMA2_CONFIG
        ),
    }

    print("\n" + "=" * 60)
    print("Summary")
    print("-------")
    for name, ok in results.items():
        print(f"{name}: {'OK' if ok else 'FAIL'}")

    if not all(results.values()):
        raise SystemExit(1)
    print(
        "\nOK: streaming PagedModel correctly truncates sliding-window attention for both "
        "Mistral (uniform) and Gemma2 (alternating), on a prompt genuinely longer than the window "
        "-- exact match, or a divergence only at a genuine baseline near-tie."
    )


if __name__ == "__main__":
    main()
