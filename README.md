# rust-llm-pager

A Rust pager that decides which parts of an LLM's KV cache stay in GPU memory and which get pushed to CPU RAM, without changing a single token of what the model generates.

As context length grows, the KV cache eventually stops fitting in VRAM. The usual fixes are capping context length or buying a bigger GPU. This project tries a third option: keep only the KV blocks that actually matter — recent tokens, high-attention tokens, or a mix of both — resident on the GPU, and let the rest live in system RAM until they're needed again. The placement decisions run in Rust; the actual tensor movement happens in Python through PyTorch.

It's still early. Validated on Qwen2, Llama, and Mistral-family models (`Qwen2.5-0.5B`/`1.5B-Instruct`, `TinyLlama-1.1B-Chat`), Linux + CUDA only, no vLLM or SGLang integration. But the core claim holds up under testing: paging KV blocks between VRAM and RAM doesn't change generation output, it cuts GPU memory used for KV cache by 98%+ at long context, and — since a custom Triton kernel replaced the reload-everything decode path — the peak GPU memory *during* a decode step now genuinely drops below an unpaged baseline's, not just between steps.

---

## Quickstart

Requires Linux, a CUDA GPU, Python 3.10+, and Rust stable. Not on PyPI yet — install from source:

```bash
git clone https://github.com/3xecut0r/rust-llm-pager.git
cd rust-llm-pager
python -m venv .venv && source .venv/bin/activate

pip install -U pip maturin
pip install torch transformers   # use a CUDA-specific torch wheel for your GPU

maturin develop -m pager/Cargo.toml --release   # build the Rust pager
pip install -e .                                 # install pager_hf
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pager_hf import PagedModel

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16,
).to("cuda")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

encoded = tokenizer("Explain KV-cache paging in one sentence.", return_tensors="pt").to("cuda")

paged_model = PagedModel(
    model,
    vram_budget=128_000_000,   # bytes of VRAM the pager may keep KV blocks in
    ram_budget=2_000_000_000,  # bytes of CPU RAM for everything else
    policy="recent_only",      # or "sinks_heavy_hitter" for short, quality-sensitive contexts
)

generated_ids = paged_model.generate(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
    max_new_tokens=64,
)
print(tokenizer.decode(generated_ids))
```

That's the whole surface for a first run. For exact dependency versions see [Installation](#installation); for persistence across turns, batching, and policy tradeoffs see [Python API: `pager_hf.PagedModel`](#python-api-pager_hfpagedmodel).

---

## Why this exists

LLM inference gets memory-bound fast as context grows. KV cache eats VRAM, and it eats more of it the longer the prompt, the more concurrent requests, or the smaller the GPU. If you're running on a 4–8GB card, this shows up quickly.

The bet here is that not all KV blocks are equally worth keeping on the GPU at any given moment. Some are ancient history the model barely attends to; a few "sink" tokens near the start matter disproportionately regardless of recency; others are genuinely load-bearing because the model keeps attending back to them. A pager that tells the difference should be able to fit a longer context in the same VRAM budget than one that just keeps everything resident, or one that naively evicts by recency alone.

Whether this belongs bolted onto an existing serving engine (vLLM, LMCache) or as its own thing is still an open question — see [Limitations](#important-limitations) for what came out of actually looking into vLLM's internals.

---

## What works today

- Rust pager core with four placement policies, exposed to Python via PyO3.
- Real GPU ↔ CPU movement of actual Qwen `past_key_values` tensors, not synthetic stand-ins.
- A persistent KV store that lives across an entire generation, growing as new blocks are produced.
- `pager_hf.PagedModel`: an installable wrapper around HuggingFace `transformers` that does all of the above behind a `generate()` call — with session persistence across calls and real batched generation.
- Every one of these is checked against a plain, unpaged baseline. If token IDs don't match exactly, it's a bug. See [How this was validated](#how-this-was-validated) for what's actually been run and what the numbers look like.

Headline result, reproducible with `python bench/real_kv_paged_model_api_mvp.py`:

```text
OK: pager_hf.PagedModel produces identical greedy generation to baseline.
```

---

## Architecture

```text
Rust pager policy
      |
      v
PyO3 Python binding
      |
      v
PyTorch KV block store
      |
      v
Real model past_key_values
      |
      v
GPU <-> CPU KV movement
      |
      v
Reconstructed cache used for generation
```

Rust decides where each block should live; Python does the actual copying. Splitting it this way means the placement logic — the part worth getting right — is pure, deterministic, and testable without a GPU, while the tensor movement stays in PyTorch where it belongs.

---

## Policies

| Policy | Description |
|---|---|
| `recent_only` | Keeps the most recent blocks in VRAM. |
| `sinks_recent` | Keeps sink blocks plus recent blocks. |
| `heavy_hitter` | Keeps blocks with high accumulated attention score. |
| `sinks_heavy_hitter` | Hybrid: sink blocks + a limited recent tail + whatever's left of the VRAM budget goes to the highest-attention blocks. |

`sinks_heavy_hitter` is the strongest policy so far, both in the multi-needle benchmark and in how much attention mass it keeps resident on GPU. `recent_only` and `sinks_recent` don't use attention scores at all for placement, which is a real tradeoff, though a smaller one than it used to be: all four policies now support batched generation (see [Batching](#batching-batch_size--1) in the [Python API](#python-api-pager_hfpagedmodel) section) — the two attention-scoring policies just cost a second Triton kernel pass to get that signal instead of reading it for free from the placement rule itself.

---

## Python API: `pager_hf.PagedModel`

Everything under `bench/` is a validation script, not the intended way to use this project. `pager_hf` is the real, installable interface — a thin wrapper around a HuggingFace causal LM that reuses the same Rust pager and KV block store, minus the debug printing and CLI plumbing that the bench scripts are full of.

`PagedModel` adapts itself to the policy you give it. `recent_only` and `sinks_recent` never look at attention scores for placement (see `pager/src/core.rs`), so no attention signal is computed for them at all — cheapest path, and the only one usable at `batch_size > 1` before this scoring extension existed (see [Batching](#batching-batch_size--1)). `heavy_hitter` and `sinks_heavy_hitter` need a real attention signal; by default (`use_streaming_attention=True`) that comes straight out of a second Triton kernel pass (`pager_hf/streaming_attention.py`), not from HuggingFace's `output_attentions`. With `use_streaming_attention=False` (the older reconstruct-and-call path, still available), `output_attentions=True` stays on during decoding instead.

All four policies scale to long context, which wasn't obvious going in. On the fallback path, every forward call processes exactly one token against the reconstructed cache, so `output_attentions=True` there only costs a `[heads, 1, total_len]` matrix per layer (linear in context length), not the quadratic `[heads, seq_len, seq_len]` a bulk forward call would need. The default streaming path avoids reconstructing the cache at all, so this cost doesn't even apply there. `bench/real_kv_paged_model_long_context_mvp.py --policy <name>` checks all four policies at ~6,000 tokens for byte-identical output against baseline:

| Policy | Same as baseline | Mean attention kept on GPU |
|---|---|---:|
| `recent_only` | `True` | n/a — placement ignores attention |
| `sinks_recent` | `True` | n/a — placement ignores attention |
| `heavy_hitter` | `True` | 0.5615 |
| `sinks_heavy_hitter` | `True` | 0.5464 |

All four keep the same fixed ~7 blocks resident on GPU (set by `vram_budget`); the attention-scored policies keep noticeably more of the real attention mass on GPU than a uniform 7-of-370 selection would (~0.019), which is the same quality edge the multi-needle benchmark shows at short context.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pager_hf import PagedModel

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    torch_dtype=torch.float16,
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
encoded = tokenizer("...", return_tensors="pt").to("cuda")

paged_model = PagedModel(
    model,
    vram_budget=128_000_000,
    ram_budget=2_000_000_000,
    policy="recent_only",  # use "sinks_heavy_hitter" for short, quality-sensitive contexts
)

generated_ids = paged_model.generate(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
    max_new_tokens=64,
)

print(paged_model.last_run_stats)  # swap volume, attention kept on GPU, etc.
```

`generate()` is greedy by default — that's what every byte-identical-to-baseline check in this README relies on. Pass `do_sample=True` for temperature/top-k/top-p sampling instead:

```python
generated_ids = paged_model.generate(
    input_ids=encoded["input_ids"],
    attention_mask=encoded["attention_mask"],
    max_new_tokens=64,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    generator=torch.Generator(device="cuda").manual_seed(0),  # optional, for reproducibility
)
```

`generator`, if given, has to be on the same device as the model — that's a PyTorch `multinomial` requirement, not something this project adds.

### Persistence across `generate()` calls

The KV block store and pager live on the `PagedModel` instance, not inside `generate()`. Each call is given the full sequence so far — previous input, previously generated tokens, and any new tokens — and only the delta beyond what was already processed gets forward-passed. That's what makes a multi-turn session cheap instead of re-prefilling from scratch on every turn:

```python
paged_model = PagedModel(model, vram_budget=128_000_000, ram_budget=2_000_000_000, policy="recent_only")

turn1 = paged_model.generate(input_ids=ids1, attention_mask=mask1, max_new_tokens=64)

# caller appends turn1's generated tokens + new user input, then continues
ids2 = torch.cat([ids1, torch.tensor([turn1], device=ids1.device), new_turn_ids], dim=1)
mask2 = torch.ones_like(ids2)
turn2 = paged_model.generate(input_ids=ids2, attention_mask=mask2, max_new_tokens=64)

paged_model.reset()  # drop the session; the next generate() call starts fresh
```

Pass a shorter `input_ids` than what's already primed, or one that diverges from the already-primed prefix, and you get a `ValueError` instead of a silently corrupted KV cache. `generate()` can extend a session or be reset — it can't rewind one.

That mutable session state also means a single `PagedModel` instance isn't meant to serve two calls at once. Calling `generate()` or `reset()` on the same instance from another thread while one is already in flight raises `RuntimeError` immediately, instead of racing on `_store`/`_tail_past` and producing quietly wrong output. Use a separate `PagedModel` per concurrent session — they don't share any state.

### Batching (`batch_size > 1`)

`generate()` accepts a real batch — one forward call per step across every row, not a Python loop over rows — under one constraint: every row needs the same total length (real tokens plus padding), and `attention_mask` per row has to be zero or more leading zeros followed by all ones — standard left-padding, the usual convention for batched causal-LM generation. Right-padding or masking with gaps in the middle raises `NotImplementedError` rather than doing something quietly wrong. This applies whether `use_streaming_attention` is on (the default) or off.

All four policies now support `batch_size > 1`, including `heavy_hitter` / `sinks_heavy_hitter`. Block placement is still one decision shared by the whole batch (one `PyPager`, one `KVBlockStore` — rows can't have independently-placed blocks), so for the two attention-scoring policies the placement score is an *average* of each row's own attention across the batch, not any single row's own preference. This never affects which tokens get generated: every row's attention always covers the full context regardless of which physical tier a block sits in — placement only changes memory residency and transfer cost, not correctness — so `same_token_ids` holds at `batch_size > 1` for every policy, the same way it does at `batch_size == 1` (`bench/streaming_paged_model_batch_verify_mvp.py`).

A block can be entirely padding for a shorter row and that's harmless, but *how* padding gets excluded from attention differs by path: the non-streaming path relies on HuggingFace's own `attention_mask` handling; the streaming path (default) computes its own `[batch, chunk_tokens]` validity mask per attention chunk from the same `attention_mask`, since the Triton kernel never sees HuggingFace's internal 4D causal mask at all. Both were checked against a per-row unpadded reference with genuinely different real lengths per row, not just different total padding amounts (`bench/streaming_paged_model_batch_verify_mvp.py`).

Position IDs are derived from `attention_mask` (`cumsum(-1) - 1`, clamped at padded positions) rather than assumed to be a plain `0..n-1` range — that's the part that actually makes padding produce correct output instead of silently wrong output.

```python
# rows can be different real lengths; left-pad to a common total length
generated = paged_model.generate(input_ids=batched_ids, attention_mask=batched_mask, max_new_tokens=64)
# -> list[list[int]] when batch_size > 1 (list[int] when batch_size == 1, unchanged)
```

`bench/real_kv_paged_model_batch_mvp.py` checks this against the ground truth: three different-length prompts, left-padded to a common length and run together in one `batch_size=3` call, produce — row for row — byte-identical output to running each prompt alone, unpadded, at `batch_size=1`.

### Observability

`pager_hf` logs through the standard `logging` module under the `pager_hf` name, so a long-lived process gets visibility without adding a metrics dependency:

```python
import logging
logging.getLogger("pager_hf").setLevel(logging.DEBUG)  # or INFO for less noise
logging.basicConfig()  # or wire your own handler/formatter
```

- `INFO`: session lifecycle — `generate()` starting a fresh session (context length, block count, policy), `reset()`, and streaming attention activating (detected `model_type`, layers patched, group size).
- `DEBUG`: per-decode-step stats — new blocks registered, GPU↔CPU bytes moved, fraction of attention mass currently GPU-resident.
- `WARNING`: CPU RAM usage crossing 80% of `ram_budget`, and a `generate()`/`reset()` call rejected because another one is already in flight on the same instance — both fire *before* whatever exception (if any) also gets raised, so they're visible even if a caller catches errors silently.
- `ERROR`: `ram_budget` actually exceeded, or `use_streaming_attention` rejected for an unrecognized architecture — logged right alongside the raised exception.

For metrics rather than logs, `paged_model.last_run_stats` after each `generate()` call is a plain `dataclasses.dataclass` (`GenerationStats`) — `dataclasses.asdict(paged_model.last_run_stats)` gives a flat dict ready to push into whatever the caller already uses (Prometheus, statsd, a plain log line), no extra dependency added on this side.

### Installing and validating

```bash
pip install -e .
```

```bash
python bench/real_kv_paged_model_api_mvp.py           # short context, sinks_heavy_hitter
python bench/real_kv_paged_model_long_context_mvp.py  # long context, recent_only
python bench/real_kv_paged_model_persistence_mvp.py   # split across generate() calls == single call
python bench/real_kv_paged_model_batch_mvp.py         # batch_size > 1 == per-row batch_size == 1
```

### Current limitations of `pager_hf`

- Greedy by default; temperature/top-k/top-p sampling is available via `do_sample=True` (see the [Python API](#python-api-pager_hfpagedmodel) section above), but no beam search.
- `batch_size > 1` needs left-padding to a common total length; right-padding or gapped masking isn't supported. Left-padding is the standard way to batch causal-LM generation anyway, so this isn't considered a gap to close. All four policies work at `batch_size > 1` now (see [Batching](#batching-batch_size--1) above).
- New tokens beyond what's already primed get fed through the model one at a time (matching how every other script in this project scores blocks per token). Priming a very long new turn in one `generate()` call isn't chunked the way the initial prefill is.
- Same VRAM ↔ CPU-only scope as the rest of the project — no SSD tier, no vLLM/LMCache integration.

---

## Does this actually save VRAM?

Everything above proves the mechanism doesn't break correctness. It doesn't prove it matters — with a 151-token prompt, the whole KV cache is a couple of MB, and paging a couple of MB around isn't solving anyone's memory problem. `bench/real_kv_vram_savings_proof_mvp.py` closes that gap: it runs at a context long enough for the KV cache itself to become a meaningful amount of memory, and measures it directly.

Getting there required fixing one thing unrelated to KV paging: a single long forward pass OOMs on its own, because HuggingFace computes logits for every position in the call, and with a ~152k vocabulary an 8,000-token prefill alone tries to allocate roughly 2.3GB just for logits. Chunked prefill — feed the prefix in pieces, discard each chunk's logits — fixes it, and both the baseline and the paged run use it identically.

This proof runs with `recent_only`, but that's not a hard requirement; see the long-context note in the [Python API](#python-api-pager_hfpagedmodel) section above for why `heavy_hitter` and `sinks_heavy_hitter` also scale. With the OOM fixed, the real result at a realistic long context:

| Context tokens | Full KV if resident on GPU | Resident on GPU under pager | Resident on CPU | Reduction | Same output as baseline |
|---:|---:|---:|---:|---:|---|
| 5,922 | 72.86 MB | 1.38 MB | 71.37 MB | 98.1% | `True` |
| 11,750 | 144.47 MB | 1.38 MB | 142.93 MB | 99.0% | `True` |

The pager keeps a fixed ~7 blocks resident on GPU (set by `VRAM_BUDGET`) no matter how long the context grows, while CPU absorbs the rest, and the generated tokens are still byte-identical to a baseline that never offloads anything. That's the actual value proposition: the same generation on a fraction of the GPU memory that would otherwise be pinned down by the KV cache, scaling as context grows instead of scaling with it.

```bash
python bench/real_kv_vram_savings_proof_mvp.py --context-tokens 12000 --new-tokens 8
```

Worth being honest about: this is still Qwen2.5-0.5B, which has only 2 KV heads — about 12KB of KV cache per token across all 24 layers, tiny even at tens of thousands of tokens. Larger models with more KV heads hit real VRAM ceilings at far shorter contexts, which is the scenario this mechanism is actually built for. This proof shows the mechanism works; it doesn't yet show a model where you'd feel the difference on real hardware.

---

## How this was validated

Getting from "a Rust struct that scores blocks" to "this doesn't break a real model's output" took a series of increasingly strict checks, each built on the last. In order:

1. **Dummy tensors first.** Before touching a real model, the pager moved 160 fake KV-shaped tensors between GPU and CPU and picked exactly the blocks it said it would keep — the pager's own list of GPU-resident blocks matched the store's actual GPU-resident blocks, byte for byte: `[0, 1, 2, 3, 20, 21, 55, 56, 146, 147, 148, 149, 150, 151, 152]` on both sides.
2. **Real Qwen KV blocks.** Same check, but with `past_key_values` actually extracted from Qwen2.5-0.5B-Instruct. Placement decisions still matched what physically ended up on the GPU.
3. **Roundtrip correctness.** KV blocks sent GPU → CPU → GPU came back bit-identical — `max_key_diff: 0.0`, `max_value_diff: 0.0`.
4. **Logits, then multi-token generation.** Reconstructed KV cache produced identical next-token logits, then identical greedy decoding across multiple tokens, then output identical to a normal, unpaged generation run.
5. **A real generation loop.** Not a one-shot check — a full multi-step decode loop paging blocks between GPU and CPU on every token produced output identical to baseline.
6. **A persistent, growing KV store.** Instead of rebuilding the store every run, one `KVBlockStore` and one `PyPager` now live across an entire generation, with new blocks appended as context grows. Validated at 24 and 64 generated tokens across all four policies:

   | Policy | Same as baseline | New blocks | Final blocks | Mean attn in VRAM | Min attn in VRAM |
   |---|---|---:|---:|---:|---:|
   | `heavy_hitter` | True | 4 | 13 | 0.8717 | 0.5137 |
   | `sinks_heavy_hitter` | True | 4 | 13 | 0.8627 | 0.6805 |
   | `recent_only` | True | 4 | 13 | 0.6244 | 0.3686 |
   | `sinks_recent` | True | 4 | 13 | 0.6244 | 0.3686 |

   `heavy_hitter` and `sinks_heavy_hitter` keep noticeably more attention mass resident in VRAM here, while all four preserve exact baseline token IDs. `real_kv_persistent_cpu_paging_loop_mvp.py` (24 tokens), `real_kv_persistent_cpu_paging_stress_mvp.py` (64 tokens, forces multiple new blocks), and `real_kv_persistent_policy_compare_mvp.py` (all four policies, `--tokens N` and `--quiet` supported) cover this.

7. **A quality benchmark, not just correctness.** A multi-needle-in-a-haystack test — 7 prompts × 15 memory-penalty settings — showed `sinks_heavy_hitter` winning 86 of 105 configurations overall, and 55 of 56 under the strictest VRAM penalties (`SSD <= -6`):

   | Policy | Wins (overall) | Wins (`SSD <= -6`) | Mean PPL | Attention in VRAM |
   |---|---:|---:|---:|---:|
   | `sinks_heavy_hitter` | 86 / 105 | 55 / 56 | ~384.3 | ~0.660 |
   | `heavy_hitter` | 8 / 105 | 0 / 56 | ~2866.9 | ~0.543 |
   | `sinks_recent` | 2 / 105 | 1 / 56 | ~3158.5 | ~0.386 |
   | `recent_only` | 9 / 105 | 0 / 56 | ~4414.3 | ~0.014 |

   The hybrid policy fixes the failure modes where pure heavy-hitter misses important recent or sink context, and where pure recency misses everything else.

8. **Does it actually matter?** All of the above ran on prompts short enough that the KV cache was a couple of MB, which proves correctness but nothing about real memory pressure. See [Does this actually save VRAM?](#does-this-actually-save-vram) above for the long-context numbers — 98–99% GPU memory reduction for KV cache, still byte-identical output.
9. **The installable API, not just internals.** Everything up to this point exercised the low-level `KVBlockStore`/`PyPager` primitives directly. `pager_hf.PagedModel` — the thing meant to actually get used — is checked separately against the same bar: identical output at short and long context, correct behavior across multiple `generate()` calls, and correct output for a real batch of different-length, left-padded prompts. See [Python API](#python-api-pager_hfpagedmodel) above.

Every one of these is a runnable script under `bench/`; none of them are CI unit tests, because they need a real GPU and a downloaded model. `python bench/real_kv_paged_model_api_mvp.py` and `python bench/real_kv_vram_savings_proof_mvp.py` are the two most worth running if you want to see the proof yourself rather than take this README's word for it.

---

## Repository layout

```text
pager/
  Cargo.toml
  Cargo.lock
  pyproject.toml   # packaging for rust-llm-pager-core, the compiled pager extension
  src/
    core.rs       # Rust pager policies and placement logic
    lib.rs        # PyO3 bindings
    metrics.rs    # Pager metrics exposed to Python

bench/
  multi_needle_soft_masked_grid_hf.py
  multi_needle_soft_masked_grid_summary.py
  multi_needle_soft_masked_grid_plot.py
  probe_needles_hf.py

  torch_kv_block_store.py
  torch_kv_offload_mvp.py
  torch_kv_pager_integration_mvp.py

  real_kv_cache_block_mvp.py
  real_kv_roundtrip_mvp.py
  real_kv_logits_roundtrip_mvp.py
  real_kv_generation_roundtrip_mvp.py
  real_kv_paged_generation_loop_mvp.py
  real_kv_paged_generation_compare_mvp.py

  real_kv_persistent_cpu_paging_loop_mvp.py
  real_kv_persistent_cpu_paging_stress_mvp.py
  real_kv_persistent_policy_compare_mvp.py
  real_kv_paged_model_api_mvp.py
  real_kv_paged_model_long_context_mvp.py
  real_kv_paged_model_persistence_mvp.py
  real_kv_paged_model_batch_mvp.py
  real_kv_paged_model_throughput_mvp.py
  real_kv_vram_savings_proof_mvp.py

  # Triton streaming-attention kernel: development history, correctness
  # anchors, and the go/no-go checkpoints that led to pager_hf/streaming_attention.py
  streaming_attention_prototype_mvp.py       # pure-PyTorch online-softmax proof, Stage 0
  streaming_attention_real_layer_mvp.py      # pure-Python per-layer patch -- the too-slow dead end
  streaming_attention_triton_kernel_mvp.py   # the Triton kernel's own dev/test file
  streaming_attention_triton_real_layer_mvp.py
  streaming_attention_full_loop_mvp.py       # full 24-layer streaming loop, peak-memory proof

  # Verification of the shipped pager_hf.PagedModel integration, not just the kernel
  streaming_paged_model_verify_mvp.py         # all 4 policies vs baseline, Qwen2
  streaming_paged_model_verify_llama_mvp.py   # same, TinyLlama-1.1B
  streaming_paged_model_verify_mistral_mvp.py # same, tiny Mistral checkpoint
  streaming_paged_model_batch_verify_mvp.py   # batch_size > 1, heterogeneous padding, all 4 policies
  streaming_paged_model_peak_memory_mvp.py    # real peak-GPU-memory reduction, streaming vs not
  streaming_group_size_sweep_mvp.py           # streaming_group_size_blocks tuning

pager_hf/
  __init__.py
  kv_block_store.py     # runtime-agnostic GPU <-> CPU tensor movement primitive
  kv_utils.py            # HuggingFace past_key_values <-> KV block conversions
  paged_model.py         # PagedModel: the installable HF integration
  streaming_attention.py # Triton streaming-attention kernels + host wrappers

tests/
  test_kv_utils.py             # CPU-only, run in CI
  test_paged_model_padding.py  # position_ids / left-padding validation, CPU-only, run in CI
  test_sampling.py             # do_sample/temperature/top-k/top-p, CPU-only, run in CI
  test_concurrency.py          # generate()/reset() lock contention + logging, CPU-only, run in CI
  test_kv_block_store.py       # GPU-only (skipped in CI, no GPU there)
  test_streaming_attention.py  # Triton kernel correctness incl. batching/padding, GPU-only (skipped in CI)

.github/workflows/
  ci.yml               # cargo test + pytest + packaging check, no GPU needed

pyproject.toml   # packaging for pager-hf
```

Two PyPI packages come out of this repo: `rust-llm-pager-core` (the compiled Rust extension, imported as `pager`) and `pager-hf` (the pure-Python `pager_hf` package, which depends on the former). They're split because a single wheel can't cleanly bundle a compiled extension under one import name alongside a separate pure-Python package under another — maturin tries to nest one inside the other when you attempt it, and splitting the packages was simpler than fighting that.

---

## Installation

This project currently expects Linux, CUDA, Python, Rust, and a working PyTorch CUDA install.

Recommended environment:

- Python 3.12
- Rust stable
- CUDA-capable GPU
- PyTorch with CUDA
- `maturin`
- `transformers`
- `triton` — required now that `use_streaming_attention=True` is the default; `pip install pager-hf` pulls it in automatically

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip maturin
pip install torch transformers triton accelerate
```

If you need a CUDA-specific PyTorch wheel, install PyTorch first according to your CUDA version. Versions used during development:

```text
torch==2.3.1+cu118
transformers==4.44.2
triton==3.6.0
```

---

## Build

Build the Rust/PyO3 extension:

```bash
maturin develop -m pager/Cargo.toml --release
```

Then check that Python can import it:

```bash
python - <<'PY'
import pager
p = pager.PyPager(
    256_000_000,
    2_000_000_000,
    64,
    8,
    0.05,
    0.20,
    "sinks_heavy_hitter",
)
print(p.metrics().tokens)
PY
```

---

## Running the demos

Compile-check everything first:

```bash
python -m py_compile bench/*.py
```

Real tensor movement (no real model, dummy KV-shaped tensors):

```bash
python bench/torch_kv_offload_mvp.py
python bench/torch_kv_pager_integration_mvp.py
```

Real Qwen KV correctness chain — extraction, roundtrip, logits, multi-token generation, paged loop, baseline comparison:

```bash
python bench/real_kv_cache_block_mvp.py
python bench/real_kv_roundtrip_mvp.py
python bench/real_kv_logits_roundtrip_mvp.py
python bench/real_kv_generation_roundtrip_mvp.py
python bench/real_kv_paged_generation_loop_mvp.py
python bench/real_kv_paged_generation_compare_mvp.py
```

Persistent CPU KV paging (accepts `--tokens N` and `--quiet`):

```bash
python bench/real_kv_persistent_cpu_paging_loop_mvp.py
python bench/real_kv_persistent_cpu_paging_stress_mvp.py
python bench/real_kv_persistent_policy_compare_mvp.py --tokens 64 --quiet
```

`pager_hf.PagedModel` itself, and the actual VRAM savings proof, are covered under [Python API](#python-api-pager_hfpagedmodel) and [Does this actually save VRAM?](#does-this-actually-save-vram) above — those are the two worth running first.

Streaming attention — the Triton kernel and its integration into `PagedModel` (see [Important limitations](#important-limitations) and [Project stage](#project-stage) for the full story):

```bash
python bench/streaming_paged_model_verify_mvp.py          # all 4 policies vs baseline, Qwen2.5-0.5B
python bench/streaming_paged_model_verify_llama_mvp.py     # same, TinyLlama-1.1B
python bench/streaming_paged_model_verify_mistral_mvp.py   # same, a tiny Mistral checkpoint
python bench/streaming_paged_model_batch_verify_mvp.py     # batch_size > 1, heterogeneous padding, all 4 policies
python bench/streaming_paged_model_peak_memory_mvp.py --context-tokens 6000 --policy sinks_heavy_hitter
python bench/streaming_group_size_sweep_mvp.py             # streaming_group_size_blocks tuning
```

---

## Tests / CI

Two tiers of correctness check, deliberately kept in different places.

Automated, in [`.github/workflows/ci.yml`](.github/workflows/ci.yml), no GPU needed:

- `black --check` and `isort --check` on `pager_hf/` and `tests/` — 120-char lines, one consistent quote style, imports sorted. Settings live in `pyproject.toml`, not scattered across CI flags.
- `cargo test` — unit tests for the Rust pager core ([`pager/src/core.rs`](pager/src/core.rs)): policy placement, VRAM budget enforcement, sink/recent pinning, `force_rebalance`, metrics. Pure logic, deterministic, no tensors involved.
- `pytest tests/` — most of it is CPU-only and actually runs in CI: `test_kv_utils.py` (block extract/reconstruct round-trip including the batch-dimension logic, tail concatenation, `past_key_values` normalization), `test_paged_model_padding.py` (`position_ids`/left-padding validation), `test_sampling.py` (temperature/top-k/top-p filtering), `test_concurrency.py` (lock contention, including the logging it now emits). Two files need a real GPU (`test_kv_block_store.py`, `test_streaming_attention.py` — the Triton kernels, batching, and padding-mask correctness) and are marked `skipif(not torch.cuda.is_available())`, so they run for real locally but are silently skipped on CI's GPU-less runners rather than failing.
- `python -m py_compile bench/*.py pager_hf/*.py` — catches syntax and import errors across everything else.
- A packaging check that builds both wheels, installs them together, and does a smoke import — the same check that caught a real bug (`pager_hf` silently missing from a wheel) during the PyPI packaging work.

Manual, local, real GPU and real model required: every `bench/real_kv_*.py` and `bench/streaming_*.py` script. These are the actual correctness and value proofs — `same_token_ids == True` against a real baseline, the VRAM savings numbers, the peak-memory reduction numbers, batch and persistence equivalence — and they need CUDA plus a downloaded model, so they don't run on free CI runners.

Run the CI-equivalent checks locally:

```bash
pip install -e ".[dev]"
black --check pager_hf/ tests/
isort --check pager_hf/ tests/
cd pager && cargo test && cd ..
python -m py_compile bench/*.py pager_hf/*.py
python -m pytest tests/ -v
```

---

## Important limitations

This is a prototype, not a production inference backend.

- Not integrated with vLLM or LMCache. Looked into what vLLM integration would actually take: its extension points (`KVConnectorBase_V1`, the pluggable `OffloadingManager`) are built for cross-request KV cache reuse — prefix caching, disaggregated prefill — not for the fine-grained, attention-driven, per-block placement within one active generation that this project does. vLLM's own memory management moves whole requests between GPU and CPU (or drops and recomputes them), not individual blocks of a live request. Worth knowing: vLLM's own CPU-offload roadmap plans round-robin then LRU eviction, not anything content-aware, so there's a real gap here — closing it would mean writing a vLLM-native plugin around this project's Rust core, not reusing `pager_hf` as-is. Parked for now, not abandoned.
- HuggingFace forward still receives a reconstructed full GPU cache each step — there's no fused kernel doing partial reconstruction, so the reconstruction cost is real and scales with block count. First measured on `Qwen2.5-1.5B-Instruct` (8-bit, GTX 1050 Ti, 4GB): at 3,000 tokens the paged run's peak GPU memory during the forward pass (3,369 MB) was actually *higher* than the unpaged baseline's peak (3,242 MB), because both `reconstruct_past_from_store` and `append_tail_to_reconstructed_past` built fresh full-cache tensors by collecting copies in a list (or a second tensor) and `torch.cat`-ing them together — the whole cache resident twice at once, twice over. Rewriting both to write directly into one pre-allocated destination tensor — sized to include the tail up front, so `append_tail_to_reconstructed_past` isn't a separate step anymore — closed the gap almost entirely: paged peak dropped to 3,241.58 MB, 0.02 MB off the baseline's 3,241.56 MB. `same_token_ids: True` throughout, zero regressions across the full test suite and bench chain. This doesn't mean paged now fits a *longer* context than baseline — the underlying limit (every token must be in the attention computation for exact correctness) is unchanged, and the pager still only saves VRAM *between* generation steps, not the peak *during* one. What it fixes is paging no longer costing *more* peak memory than not paging at all. Going below baseline's peak needs incremental cache *growth* (skip rewriting blocks that didn't change) or a real paged-attention kernel (à la vLLM) — see [Project stage](#project-stage).
- No SSD tier. Anything not on GPU currently lives in CPU RAM; a colder third tier isn't implemented. `ram_budget` is enforced now: exceeding it raises `RuntimeError` with a clear message on the offload that would go over, instead of silently growing CPU RAM without limit (previously the Rust pager's logical "SSD" tier and its "RAM" tier were both just placed on CPU with no cap at all — `ram_budget` did nothing on the Python side). It's still a hard stop, not an SSD fallback: there's nowhere further to spill to yet.
- Real-model demos originally used only `Qwen/Qwen2.5-0.5B-Instruct` (2 KV heads, tiny KV cache even at long context). Since validated on `TinyLlama-1.1B-Chat` (fp16) and `Qwen2.5-1.5B-Instruct` (8-bit via `bitsandbytes`) on an actual 4GB GTX 1050 Ti — both reproduce the same byte-identical output and 100% GPU-resident KV reduction between steps, but this is also what surfaced the peak-memory limitation above.
- The Rust pager uses a fixed logical block size (16MiB) for its own placement math, independent of how large a real tensor block actually is; the Python side separately reports real tensor bytes moved. Documented, not a bug, but worth knowing if you're trying to reconcile the two sets of numbers.
- `bench/real_kv_paged_model_throughput_mvp.py` measures steady-state decode speed. This used to be a real cost: with the old reconstruct-and-call path (still available via `use_streaming_attention=False`), paged decode ran about 1.17-1.18x slower per token than the unpaged baseline on `Qwen2.5-0.5B` (GTX 1050 Ti) — reloading every block back to GPU and rebuilding the full cache every step, whether or not that step's attention actually needed it. With streaming attention now the default, that reload is gone entirely and the Triton kernel itself is faster than dense attention (see below), so paged decode is now *faster* than baseline, not slower: 6.46 vs 3.02 tok/s at 2,000 tokens (0.47x of baseline's time, i.e. ~2.1x the throughput). Still one small model on one weak GPU, not a production throughput/latency characterization, and nothing here has been measured against a real serving workload (concurrent requests, varied prompt lengths, a real scheduler).
- Closed the peak-memory ceiling above with a real custom Triton kernel, wired into `pager_hf.PagedModel` and on by default — this is the one item on this list that moved from "limitation" to "done." The mechanism is streaming, block-by-block attention using the online-softmax algorithm (the exact reorganization FlashAttention uses, not an approximation). A pure-PyTorch prototype (`bench/streaming_attention_prototype_mvp.py`) proved the math exact first; hooking it into one real `Qwen2Attention` layer via monkey-patching (`bench/streaming_attention_real_layer_mvp.py` — the only extension point `transformers==4.44.2` offers, no attention-implementation plugin API before ~4.48) matched stock output exactly but was ~5x slower for *one* layer alone than the entire current 24-layer step: a pure-Python per-block loop pays kernel-launch overhead on every tiny op. Writing an actual Triton kernel fixed that — verified `tl.dot` and the alternative flash-decoding-style broadcast-`tl.sum` pattern both work correctly on this GTX 1050 Ti (Pascal) despite older discussions claiming Pascal's `tl.dot` is broken; the right primitive for this project's always-`query_len=1` decode step is the broadcast-`tl.sum` one anyway (matrix-vector, not matrix-matrix). One layer's Triton attention over the full context runs about 10x *faster* than the entire current 24-layer step, not just "not slower." The shipped kernel module (`pager_hf/streaming_attention.py`) also computes per-pager-block attention mass (a second pass, with the online-softmax stats fixed from the first) so `heavy_hitter`/`sinks_heavy_hitter` — which score blocks by attention, not just recency — get a real signal instead of needing `output_attentions=True`. `PagedModel` patches every decoder layer to stream historical blocks from the `KVBlockStore` in small groups (`streaming_group_size_blocks`, default 64 — see the note on group size below) instead of reconstructing the whole context into one GPU-resident cache first; checked token-exact against both an unpaged baseline and the reconstruct-and-call path for all four policies, with real measured peak-memory reduction on the integrated product itself (not just the `bench/` prototype): 8.5% at 3,000 tokens, 16.2% at 5,922 tokens, growing with context length. (vLLM itself was considered as a way to skip writing a kernel at all, reusing its own fast PagedAttention plus this project's Rust-scored content-aware placement on top — dead end on this hardware: vLLM hard-requires compute capability ≥7.5 and does not run on Pascal at all.)
- `use_streaming_attention` now also supports `batch_size > 1` for every policy, including `heavy_hitter`/`sinks_heavy_hitter` (the placement score becomes an average across batch rows — see [Batching](#batching-batch_size--1) — which is fine since placement never affects generated tokens, only memory residency), and Qwen2-, Llama-, and Mistral-family decoder layers — verified end to end on `TinyLlama-1.1B-Chat` and a real (if tiny, randomly-initialized) Mistral checkpoint in addition to `Qwen2.5-0.5B-Instruct`, including a batch with genuinely different real content lengths per row (`bench/streaming_paged_model_batch_verify_mvp.py`, `bench/streaming_paged_model_verify_llama_mvp.py`, `bench/streaming_paged_model_verify_mistral_mvp.py`). The model families turned out to differ in more than attribute names: in `transformers==4.44.2`, Qwen2's rotary embedding still uses the older `rotary_emb(x, seq_len=N)` → full-table-then-index convention, while Llama's and Mistral's have already moved to `rotary_emb(x, position_ids)` → pre-indexed cos/sin — Llama's decoder layer additionally precomputes it once and shares it across layers via a `position_embeddings` kwarg that Mistral's decoder layer doesn't have at all (Mistral calls `rotary_emb` itself inside `self_attn`, same as the fallback path already needed). Getting any of this wrong doesn't crash — it silently computes a plausible-looking wrong rotation — so every convention is handled explicitly (`_compute_rope` in `paged_model.py`) rather than assumed compatible; other architectures raise `NotImplementedError` instead of guessing. A `bench/streaming_group_size_sweep_mvp.py` sweep across group sizes 4-128 found peak memory essentially flat in that range on this model (the gather buffer is tiny next to the model's own footprint) while wall-clock time dropped meaningfully with larger groups (fewer kernel launches) — 64 is past the point of diminishing returns there and is now the default, up from an untuned 16.
- Structured logging via the standard `logging` module (`logging.getLogger("pager_hf")`), for anyone running this in a long-lived process who wants visibility without instrumenting it themselves. `INFO` logs session lifecycle (start, reset, streaming-attention activation with the detected `model_type`); `DEBUG` logs per-step transfer/placement stats; `WARNING` fires once CPU-resident bytes cross 80% of `ram_budget` (see `_RAM_BUDGET_WARNING_THRESHOLD` in `kv_block_store.py`), before the hard `RuntimeError` at 100%; contended `generate()`/`reset()` calls and rejected offloads log too, not just raise. No metrics-backend dependency added — wire the log records or `GenerationStats` (already a plain dataclass, `dataclasses.asdict()`-able) into whatever the caller already uses.

---

## License

Apache-2.0. See [LICENSE](LICENSE).

---

## Project stage

Research prototype, not production software.

Current milestone: a real, installable API (`pager_hf.PagedModel`) that pages KV cache for any of the four policies, at both short and long context, with session persistence, batched generation, temperature/top-k/top-p sampling, an enforced `ram_budget`, and safe concurrent-call rejection — all checked byte-identical against an unpaged baseline (for the greedy default), plus a Rust/Python test suite in CI.

The peak-memory ceiling that used to sit here as the "real next problem" (see [Limitations](#important-limitations)) is now solved in the shipped product, not just `bench/`, and on by default: a real custom Triton kernel (`pager_hf/streaming_attention.py`), online-softmax streaming attention that carries its accumulator state across separate kernel launches, gathering only a small group of blocks into GPU memory at a time instead of the whole context. All four policies are supported, including the two that score blocks by attention (`heavy_hitter`, `sinks_heavy_hitter`) via a second Triton pass that computes per-block attention mass directly, without needing HuggingFace's own `output_attentions=True`. Measured on `Qwen2.5-0.5B-Instruct`, on the integrated `PagedModel` itself (`bench/streaming_paged_model_peak_memory_mvp.py`), with a constrained `vram_budget` so blocks genuinely offload between steps: 8.5% peak-memory reduction during the decode step at 3,000 tokens, 16.2% at 5,922 tokens — growing with context length, as it should — with `same_token_ids: True` against both an unpaged baseline and the reconstruct-and-call path, across all four policies (`bench/streaming_paged_model_verify_mvp.py`). The same kernel is also just fast enough to flip the project's own throughput number: one layer's attention over the full context runs about 10x faster than the entire current 24-layer step, and with the old per-step reload gone too, paged decode went from ~1.17x *slower* than baseline to ~2.1x *faster* (see [Limitations](#important-limitations)). vLLM was considered as a shortcut (reuse their kernel, add this project's Rust-scored placement on top) and ruled out on this exact hardware: it hard-requires compute capability ≥7.5, and the GTX 1050 Ti (6.1) simply isn't in range — confirmed by a closed upstream PR, not just documentation.

What used to be opt-in caveats are now closed too: `use_streaming_attention` (now the default) supports `batch_size > 1` for *every* policy, including `heavy_hitter`/`sinks_heavy_hitter`, and Qwen2-, Llama-, and Mistral-family decoder layers — verified on `TinyLlama-1.1B-Chat` and a real Mistral checkpoint in addition to `Qwen2.5-0.5B-Instruct` (see [Limitations](#important-limitations) for what actually differs between the RoPE conventions under the hood). The group size that bounds the gather buffer was swept (`bench/streaming_group_size_sweep_mvp.py`) and bumped from an untuned 16 to 64. Structured logging was also added (`logging.getLogger("pager_hf")`) so a long-lived process has some visibility into session lifecycle, per-step transfers, and `ram_budget` pressure without needing its own instrumentation.

Next: extend `use_streaming_attention` to architectures beyond Qwen2/Llama/Mistral — unscoped work, not a proven blocker. Real production readiness beyond that is a different, much bigger question than anything on this list: no integration with a serving framework (no continuous batching, no request scheduler — one `PagedModel` is one session), and no beam search. Those aren't scoped-down versions of what's here; they're separate undertakings on top of it, not "finish the library." Then: publish `rust-llm-pager-core` and `pager-hf` to PyPI — packaging is done and tested locally, just not uploaded yet.
