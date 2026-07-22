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

### Beam search

`generate_beam_search` keeps `num_beams` parallel hypotheses of *one* prompt and, at every step, picks the globally best `num_beams` continuations across all beams combined — not the best continuation per row independently, which is what `generate()` does for a real batch. That's a genuinely different selection rule (a strong beam can spawn more than one child next step; a weak one can die entirely), so it drives its own decode loop rather than reusing `generate()`'s — but reuses the exact same single-token-forward-plus-tier-bookkeeping method underneath (`_forward_step`), unchanged.

```python
paged_model = PagedModel(model, vram_budget=128_000_000, ram_budget=2_000_000_000, policy="sinks_heavy_hitter")
best_ids = paged_model.generate_beam_search(
    input_ids=ids, attention_mask=mask, num_beams=4, max_new_tokens=32, eos_token_id=tokenizer.eos_token_id
)
```

`input_ids` can be a batch of independent prompts too (`batch_size > 1`) — each one runs its own `num_beams`-wide search independently; candidates are never mixed across different prompts (top-*k*, or sampling under `do_sample`, happens separately per prompt group). `length_penalty` (default `1.0`, matching HuggingFace's own default) divides a candidate's cumulative log-probability by its length at final-selection time only, not during the per-step search itself — only `length_penalty=0.0` (pure cumulative log-probability, no normalization at all) is verified byte-exact against HuggingFace's reference; nonzero values change the final choice in the expected direction but aren't guaranteed bit-for-bit identical to HuggingFace's own separate per-step hypothesis bookkeeping. `num_return_sequences` (1..`num_beams`) returns more than one ranked candidate per prompt. `do_sample=True` turns this into "stochastic beam search" — each step samples `num_beams` continuations from the (temperature/top-*k*/top-*p*-filtered) joint candidate distribution instead of taking the deterministic top-*k*; a real but much rarer hybrid, checked only for valid, non-crashing output rather than exact agreement with HuggingFace (the two implementations' RNGs don't match, so sampled tokens are expected to differ). A beam that emits `eos_token_id` is "frozen" (forced to keep emitting `eos_token_id` at zero further score change) rather than dropped, so it stays a valid candidate without artificially shrinking the beam count mid-search. Doesn't support the `generate()` persistence-across-calls contract — every call starts a fresh session.

Return shape follows `generate()`'s own `batch_size` convention: `list[int]` at `batch_size == 1, num_return_sequences == 1`; `list[list[int]]` if either one is `> 1`; `list[list[list[int]]]` if both are.

Reordering a beam's whole KV-cache history into a new row position (sometimes duplicating it, sometimes dropping it) is a single `KVBlockStore.reorder_batch_rows(new_row_indices)` call — the block store already keeps every row's data as a plain tensor dimension (`[num_layers, batch, block_len, kv_heads, head_dim]`), so this is just re-indexing that dimension with a Python list of parent indices, not a new storage mechanism. A batch of independent prompts just means more rows in that same dimension (`batch_size * num_beams` total), with top-*k* selection reshaped to operate per prompt group rather than across the whole batch.

**Verified against HuggingFace's own reference, not just internal self-consistency**: the actual correctness bar here is agreement with `model.generate(num_beams=N, ...)`, *not* `same_token_ids` against greedy — beam search is expected to sometimes diverge from greedy, that's the point of it. `bench/real_kv_beam_search_mvp.py` matches HuggingFace's beam search output exactly across five real scenarios: `length_penalty=0.0` (both `use_streaming_attention=True` and `False`), the new `length_penalty=1.0` default, `num_return_sequences=2`, and a batch of 2 independent prompts — on a real generation that genuinely exercises beam death and duplication (not a trivial single-beam case), plus a `do_sample=True` sanity check. One real bug caught before the first of these matched: every beam starts as an identical copy of the same prompt, so if every beam's score started at 0, the first step's top-*k* would pick the *same* best token from several duplicate rows instead of the true top-`num_beams` distinct tokens from one distribution — fixed by the standard beam-search initialization (only the first beam of each prompt group starts active at score 0, every other beam starts at `-inf`, so only one real distribution can contribute candidates on the first step).

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

### Serving multiple concurrent requests

`pager_hf.serving` wraps `PagedModel` in a small FastAPI server so more than one request can be in flight without each one blocking the next until it fully finishes. Install the extra and run it:

```bash
pip install "pager-hf[serve]"
python -m pager_hf.serving --model Qwen/Qwen2.5-0.5B-Instruct \
    --total-vram-budget-mb 2000 --total-ram-budget-mb 8000 --max-concurrent-sessions 4
```

```bash
curl http://localhost:8000/v1/completions -d '{"prompt": "Once upon a time,", "max_tokens": 32}'
```

**How requests actually get scheduled, in order of preference.** Every active session takes its own turn at least once per round — nothing ever blocks waiting for another request to fully complete. A brand-new request's *first* step (prefill + first decode token) always goes through its own `PagedModel.generate()` call, since batching prefills of different-length prompts together is a different, harder problem this doesn't attempt. From the second step on, sessions that currently share the same tail length (they naturally converge to shared tail lengths over time, since every step grows every active tail by exactly one token in lockstep) are advanced together via **`pager_hf.batched_decode.batched_decode_step`** — real vLLM-style batched continuous batching, several sessions' forward passes combined into one `model.forward()` call, not just interleaving. Any already-started session without a tail-length partner this round falls back to its own `generate()` call instead of waiting for one. This all requires `use_streaming_attention=True` (the default); with it off, every step falls back to plain per-session calls.

Each request gets its own `PagedModel` session (cheap to construct, no GPU work until first used) with an equal share of the configured total VRAM/RAM budget across `--max-concurrent-sessions` — a simple, conservative first cut, not adaptive to actual per-session usage.

**Multi-turn conversations.** Pass no `session_id` and every call is exactly the one-shot completion above — a fresh `PagedModel`, thrown away when the call finishes. Pass a `session_id` (from `POST /v1/sessions`) instead, and `pager_hf.serving` keeps that request's `PagedModel` — and its KV cache — alive across calls, so each `/v1/completions` call only sends the *new* turn's text, not the whole conversation resent from scratch:

```bash
session_id=$(curl -s http://localhost:8000/v1/sessions -X POST | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

curl http://localhost:8000/v1/completions -d "{\"prompt\": \"Once upon a time,\", \"max_tokens\": 32, \"session_id\": \"$session_id\"}"
curl http://localhost:8000/v1/completions -d "{\"prompt\": \" And then,\", \"max_tokens\": 32, \"session_id\": \"$session_id\"}"

curl http://localhost:8000/v1/sessions/$session_id -X DELETE
```

This is exactly `PagedModel.generate()`'s own existing session persistence (`_extend_session`, see [Persistence across `generate()` calls](#persistence-across-generate-calls)), just exposed over HTTP: each new turn's tokens are appended to the session's growing sequence server-side, and only the new tail gets forward-passed. No chat templating is applied — a turn's `prompt` is tokenized and appended as-is, so a caller wanting chat-style turns supplies its own role markers. A persistent session and the one-shot round-robin pool currently draw from the same `--max-concurrent-sessions`-sized budget slice but are capped **independently** (not by one shared counter) — a client holding many sessions open while also issuing many one-shot calls can, in this first pass, exceed the intended total VRAM/RAM budget. There is also no idle-timeout eviction yet: `DELETE /v1/sessions/{id}` is the only way to free a session, so a client that opens sessions and never deletes them will eventually exhaust the pool (`POST /v1/sessions` then 503s).

Verified on a real model: `bench/serving_concurrent_requests_mvp.py` checks a request served through the scheduler produces token-identical output to calling `PagedModel.generate()` directly (same correctness bar as everywhere else in this project), a short request submitted alongside a long one completes without waiting for the long one to finish (genuine interleaving), that the batched path actually fires for 3 real concurrent same-prompt requests submitted together (instrumenting the scheduler's own batched-step method directly, not just trusting the design), all three still matching the isolated reference exactly, *and* that two `/v1/completions` calls sharing one `session_id` — the second call's prompt a genuinely new piece of text, not a repeat — produce the exact same two turns of output as one `PagedModel` session's `generate()` called twice in a row, proving the KV cache really carries over between HTTP calls and not just that the endpoint accepts a `session_id` without crashing.

**Operational tooling.** Beyond structured logging, `pager_hf.serving` now has:

- **Health check** — `GET /healthz`, always open even when auth is configured. Returns `{"status", "thread_alive", "active_requests", "queued_requests", "active_sessions"}`, HTTP 503 when `thread_alive` is `False` — the one silent-failure mode this design is prone to: if the scheduler's background thread ever dies from an unhandled exception, the HTTP layer would otherwise keep looking fine while nothing is actually being processed.
- **Metrics** — `GET /metrics`, Prometheus text exposition format (`prometheus_client`; one `CollectorRegistry` per scheduler by default, or one shared registry across several — see multi-GPU below — with every metric always carrying a `device` label so they never collide). Tracks `pager_hf_requests_total{outcome,device}`, `pager_hf_tokens_generated_total{device}`, `pager_hf_decode_round_seconds{device}`, `pager_hf_step_calls_total{mode="single"|"batched",device}` (real, always-on visibility into whether Stage 2 batching actually engages, not just an ad hoc test instrumentation), and gauges for active/queued requests and active sessions.
- **Auth** — opt-in, shared-secret only: pass `--api-key` (or set `PAGER_HF_API_KEY`) and every route except `/healthz` requires `Authorization: Bearer <key>`; omit it and every route stays open, exactly as before this existed. Not real multi-user auth (no per-key identity, scoping, or rotation) — just enough to keep the server off the open network.
- **Graceful shutdown** — `ContinuousBatchingScheduler.stop(timeout=...)` now actually drains: it stops admitting new requests but lets everything already active finish naturally before the background thread exits, force-failing anything that doesn't finish within the timeout instead of leaving it to hang forever. `python -m pager_hf.serving` wires this to SIGTERM/SIGINT via `--shutdown-timeout` (default 30s) — confirmed on a real server: a 40-token request in flight when SIGTERM arrived completed in full before the process exited.
- **Backpressure** — `--max-queue-depth` (default 100) bounds how much work can be accepted but not yet running; once hit, `/v1/completions` returns HTTP 429 with `Retry-After` instead of queuing indefinitely with no feedback to the client.

Verified with new CPU-only tests (`tests/test_serving_scheduler.py`, `tests/test_serving_app.py`) covering the drain/force-retire paths, capacity rejection, health status, and auth/error-to-status-code mapping, plus a real-traffic check in `bench/serving_concurrent_requests_mvp.py` that scrapes `/metrics` after the checks above and confirms the counts reflect the actual requests just served.

**Multiple GPUs, one process.** `--devices` takes a comma-separated list (default `cuda:0`, today's exact single-GPU behavior) — one full model replica plus its own `ContinuousBatchingScheduler` per device:

```bash
python -m pager_hf.serving --model Qwen/Qwen2.5-0.5B-Instruct \
    --devices cuda:0,cuda:1 --total-vram-budget-mb 2000 --total-ram-budget-mb 8000 --max-concurrent-sessions 4
```

This is **data-parallel replication** (N independent copies of a model that already fits on one GPU), not model/tensor parallelism (splitting one model too large for a single GPU across several) — the latter is a different, harder problem this doesn't attempt, consistent with this project's standing constraint that a model must already fit by weights. `--max-concurrent-sessions`/`--max-queue-depth`/`--total-vram-budget-mb`/`--total-ram-budget-mb` are all **per GPU**, not split across `--devices`.

A new `pager_hf.serving.multi_gpu.MultiGpuScheduler` sits in front of the per-GPU schedulers and duck-types the exact same interface `create_app` already calls, so nothing in the HTTP layer branches on single- vs. multi-GPU. One-shot `submit()` calls and new sessions route to whichever GPU currently has the least work queued or running; a multi-turn session stays on the GPU it started on for its whole lifetime (its `PagedModel`/KV cache lives there, and sessions are never migrated for load reasons). `/healthz` and `/metrics` aggregate across every GPU (`/healthz` also reports a per-GPU `gpus` breakdown; unhealthy if *any* one GPU's scheduler thread has died).

**Scope, stated plainly:** this is horizontal scaling *within one machine*. Scaling across multiple machines/processes is a reverse-proxy/orchestrator's job (nginx, envoy, a Kubernetes Service) routing to N independent `pager_hf.serving` processes using the `/healthz`/`/metrics` endpoints already built here — not reinvented in this library.

Routing/session-affinity/health-aggregation logic is verified with CPU-only fakes (`tests/test_multi_gpu_scheduler.py`): least-loaded routing, session-capacity fallback when the least-loaded GPU's *own* session registry happens to be full, session affinity even when another GPU becomes less loaded later, and aggregated health/unhealthy-if-any-one-GPU-is-down. The device-placement refactor this needed (tokenization now stays on CPU; whichever scheduler a request lands on moves it to its own device — see `ContinuousBatchingScheduler`'s `device` parameter) was re-verified end to end on the single real GPU available in dev (`bench/serving_concurrent_requests_mvp.py`, byte-identical results to before the refactor). **Not yet verified on real multi-GPU hardware** — that needs a rented 2+ GPU instance, which wasn't available at the time this was built; flagged as the next concrete step, not assumed to work from the unit tests alone.

### Batched cross-session decode

The mechanism `pager_hf.serving` uses internally, described above.

`pager_hf.batched_decode.batched_decode_step(sessions, next_input_ids, device)` advances several independent `PagedModel` sessions by one token each in a **single combined `model.forward()` call**. Each session keeps its own `KVBlockStore`/tail cache/Rust pager (only the *compute* is shared, not the memory-placement decisions); a monkey-patched layer forward gathers each combined-batch row's K/V from *that row's own* store, padding sessions with fewer historical blocks with zero-and-invalid-masked blocks — reusing the same `valid`-mask mechanism that already excludes a shorter row's padding within one session's own `batch_size > 1` call, just applied across sessions instead of within one. `pager_hf.serving`'s scheduler is the intended caller (see above), but it's a standalone function usable directly:

```python
from pager_hf.batched_decode import batched_decode_step

sessions = [session_a, session_b, session_c]  # each already primed via its own generate(..., max_new_tokens=0)
next_tokens = [last_token_a, last_token_b, last_token_c]
logits = batched_decode_step(sessions, next_tokens, device)  # one forward call, not three
next_tokens = [int(row.argmax()) for row in logits]
```

Scope, each enforced explicitly rather than silently assumed: every session must be single-row (`batch_size == 1`), `use_streaming_attention=True`, share the same underlying model/`tokens_per_block`/`streaming_group_size_blocks`, use the same kind of policy (attention-scoring or not — they drive different kernel passes), and currently have the *same tail length* (a caller groups sessions by tail length per round; `pager_hf.serving`'s scheduler does exactly this). Sessions at *different* history lengths in whole blocks batch together fine — that's the actual point of the mechanism, exercised for real, not just the trivial equal-length case. Sliding-window architectures (Mistral, Gemma2) are explicitly rejected here for now — batched rows can be at different absolute positions, so the window bound would need to be per-row, not the single scalar the single-session path uses; `PagedModel.generate()` directly (including Stage 1 round-robin serving) is unaffected.

Verified on a real model (`bench/real_kv_batched_decode_mvp.py`): three sessions of *different* real history lengths (0, 1, and 2 full blocks), advanced together, each match their own isolated `PagedModel.generate()` run exactly — for both `sinks_heavy_hitter` (the two-pass, mass-tracking kernel path) and `recent_only` (the simple one-pass path). A real, measured **1.25-1.3x speedup** over pure round-robin on the dev GPU with just 3 sessions. One real numerical bug caught and fixed along the way, in the shared Triton kernel itself (`pager_hf/streaming_attention.py`, affects the single-session path too, just never triggered there): when a session has zero real history and its *very first* KV group is batched alongside sessions that do have history, the online-softmax update computes `-inf - (-inf)` (both the running max and the new chunk's max are `-inf`, since nothing valid has been seen at all yet) — an indeterminate form that evaluates to `NaN` in IEEE float arithmetic and silently poisons that session's entire output. Fixed by treating the "nothing valid seen at all so far" case explicitly as a zero contribution instead of computing the indeterminate exponential.

**Re-verified at a scale that actually matters** (`bench/real_kv_batched_decode_scale_mvp.py`, rented A40, real `Qwen2.5-7B-Instruct`, `bfloat16`): **8** concurrent sessions spread across 0 to 7 real history blocks — the 3-session dev-GPU proof above was a toy compared to this. 7 of 8 matched their isolated reference byte-for-byte; the 8th diverged at the second generated token with a `top1_vs_top2` logit gap of exactly `0.0000` — a genuine bf16 near-tie (batched vs. single-row matmuls can round differently due to a different summation order for mathematically identical work, a well-known property of batched GPU matmul, not a bug specific to this codebase), reproduced identically across two separate runs. All 7 *other* sessions — spanning the full 0–7 block range that exercises the cross-session padding mechanism — matched exactly, so this isn't the padding logic misbehaving; it's ordinary floating-point non-determinism surfacing on randomly-generated (not real, coherent) token histories, which are unusually prone to near-ties in the first place. Throughput at this real scale: **2.9-3.0x faster than round-robin** (0.82-0.84s vs 2.4s for 6 decode steps across 8 sessions) — a bigger win than the 3-session dev-GPU number, as expected (more sessions per batched call amortizes the fixed cost of a forward pass further).

**Short sessions no longer pay full compute for a long session's padding.** Until now, `batched_decode_step` padded every session's history to the round's longest session (`max_blocks`) and relied on the existing `valid` mask to zero out padding's *contribution* — but the Triton kernel still *ran* the same number of loop iterations for every row regardless of how much of that was real data, so a 1-block session batched next to a 50-block one did as much work as if it also had 50 blocks. Every kernel program here is already one-per-row (grid = `batch * num_kv_heads`), so the fix is a genuine per-row loop bound instead of one shared scalar: each program now reads its own `active` flag and skips a KV group's loop entirely (zero iterations) when it has no real data anywhere in that group, rather than iterating the whole thing only to have every position masked out anyway. The one group per row where its own real/padding split falls inside the group (real data is a suffix, since padding is always a left-padded prefix) still runs in full, exactly as before — a small, bounded, non-scaling residual, not the full vLLM-style zero-waste design (that needs `KVBlockStore`'s storage model itself to change, a separate, larger undertaking not attempted here). Verified as a true no-op via a GPU-gated unit test (a forced-inactive row with a fully valid chunk is byte-identical to its pre-call state, while other rows in the same launch are unaffected) and re-checked against the existing real-model correctness bars (`bench/real_kv_batched_decode_mvp.py`, `bench/real_kv_batched_decode_scale_mvp.py` both still match token-for-token). A new `bench/real_kv_batched_decode_skew_mvp.py` measures the actual point directly — adding one 1-block session to a batch of four 24-block sessions costs **~0.34x** of an average long session's own cost on the dev GPU, not the ~1.0x it would cost without the skip; a bigger, more decisive gap is expected at real scale (rented A40, larger block-count skew), not yet re-measured there.

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

- Greedy by default; temperature/top-k/top-p sampling is available via `do_sample=True`, and beam search via `generate_beam_search` (see the [Python API](#python-api-pager_hfpagedmodel) section above) — supports a batch of independent prompts, `length_penalty`, `num_return_sequences`, and (as a sanity-checked, not HuggingFace-exact-matched, hybrid) `do_sample`.
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
  streaming_paged_model_verify_new_archs_mvp.py # same, tiny Qwen2MoE/Starcoder2/Gemma/Phi3/Gemma2 checkpoints
  streaming_paged_model_verify_sliding_window_mvp.py # real sliding-window truncation, tiny Mistral/Gemma2 w/ a small window
  streaming_paged_model_batch_verify_mvp.py   # batch_size > 1, heterogeneous padding, all 4 policies
  streaming_paged_model_peak_memory_mvp.py    # real peak-GPU-memory reduction, streaming vs not
  streaming_group_size_sweep_mvp.py           # streaming_group_size_blocks tuning
  streaming_paged_model_scale_test_mvp.py     # real-scale test: 7B+ model, real long context, rented GPU
  streaming_attention_kernel_scaling_probe.py # kernel-only vs dense attention timing at real model shapes
  streaming_gather_vs_kernel_profile_mvp.py   # where real decode-step time actually goes (gather vs kernel)

  serving_concurrent_requests_mvp.py          # Stage 1 serving layer: correctness + genuine interleaving proof
  real_kv_batched_decode_mvp.py               # Stage 2 core mechanism: batched cross-session decode, correctness + throughput
  real_kv_batched_decode_scale_mvp.py         # same, at real scale: 8 sessions, 7B model, rented A40
  real_kv_batched_decode_skew_mvp.py          # per-row padding-skip optimization: short session's marginal cost vs a long one's
  real_kv_paged_model_short_prompt_mvp.py     # regression: short (sub-tokens_per_block) prompts, all 4 policies
  real_kv_beam_search_mvp.py                  # generate_beam_search vs HuggingFace's own beam search reference

pager_hf/
  __init__.py
  kv_block_store.py     # runtime-agnostic GPU <-> CPU tensor movement primitive
  kv_utils.py            # HuggingFace past_key_values <-> KV block conversions
  paged_model.py         # PagedModel: the installable HF integration
  streaming_attention.py # Triton streaming-attention kernels + host wrappers
  serving/               # optional FastAPI serving layer (pip install pager-hf[serve])
    scheduler.py          # round-robin decode scheduler (Stage 1 continuous batching)
    app.py                # FastAPI app / v1/completions endpoint
    __main__.py            # python -m pager_hf.serving CLI

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
python bench/streaming_paged_model_verify_new_archs_mvp.py # same, tiny Qwen2MoE/Starcoder2/Gemma/Phi3/Gemma2 checkpoints
python bench/streaming_paged_model_verify_sliding_window_mvp.py # real sliding-window truncation, tiny Mistral/Gemma2
python bench/streaming_paged_model_batch_verify_mvp.py     # batch_size > 1, heterogeneous padding, all 4 policies
python bench/streaming_paged_model_peak_memory_mvp.py --context-tokens 6000 --policy sinks_heavy_hitter
python bench/streaming_group_size_sweep_mvp.py             # streaming_group_size_blocks tuning

# Real scale, needs a bigger GPU than the 4GB dev card and downloads a 7B+ model:
python bench/streaming_paged_model_scale_test_mvp.py --model Qwen/Qwen2.5-7B-Instruct --context-tokens 8000 --dtype bfloat16
```

The serving layer (see [Serving multiple concurrent requests](#serving-multiple-concurrent-requests)):

```bash
pip install "pager-hf[serve]"
python bench/serving_concurrent_requests_mvp.py   # token-exact correctness + genuine interleaving, real model
python bench/real_kv_batched_decode_mvp.py         # Stage 2 core mechanism: correctness + measured speedup vs round-robin
python bench/real_kv_batched_decode_skew_mvp.py    # per-row padding-skip: short session's marginal cost vs a long one's

# Real scale, needs a bigger GPU than the 4GB dev card and downloads a 7B+ model:
python bench/real_kv_batched_decode_scale_mvp.py   # 8 sessions, Qwen2.5-7B-Instruct, ~2.9-3.0x speedup
```

Beam search (see [Beam search](#beam-search)):

```bash
python bench/real_kv_beam_search_mvp.py            # generate_beam_search vs HuggingFace's own beam search reference
```

---

## Tests / CI

Two tiers of correctness check, deliberately kept in different places.

Automated, in [`.github/workflows/ci.yml`](.github/workflows/ci.yml), no GPU needed:

- `black --check` and `isort --check` on `pager_hf/` and `tests/` — 120-char lines, one consistent quote style, imports sorted. Settings live in `pyproject.toml`, not scattered across CI flags.
- `cargo test` — unit tests for the Rust pager core ([`pager/src/core.rs`](pager/src/core.rs)): policy placement, VRAM budget enforcement, sink/recent pinning, `force_rebalance`, metrics. Pure logic, deterministic, no tensors involved.
- `pytest tests/` — most of it is CPU-only and actually runs in CI: `test_kv_utils.py` (block extract/reconstruct round-trip including the batch-dimension logic, tail concatenation, `past_key_values` normalization, and the `num_blocks == 0`/short-prompt edge case), `test_paged_model_padding.py` (`position_ids`/left-padding validation), `test_sampling.py` (temperature/top-k/top-p filtering), `test_concurrency.py` (lock contention, including the logging it now emits), `test_serving_scheduler.py` (round-robin admission/interleaving/EOS-retirement logic, the draining `stop()`/force-retire timeout path, and backpressure rejection, all against a fake `PagedModel` stub), `test_serving_app.py` (the HTTP layer against a fake scheduler: auth gating, health/metrics response shape, and error-to-status-code mapping for every exception type `scheduler.py` can raise — no real model or GPU needed for any of this, CI installs `fastapi`/`prometheus_client`/`httpx` just for these two files), `test_multi_gpu_scheduler.py` (`MultiGpuScheduler`'s own routing logic against fake per-GPU scheduler stubs: least-loaded routing, session-capacity fallback, session affinity, aggregated health), `test_batched_decode.py` (Stage 2's validation checks — mismatched model/`tokens_per_block`/tail length/policy-kind, missing active session — against lightweight `PagedModel` instances with directly-set session state, no CUDA needed since these all run before any real computation). Two files need a real GPU (`test_kv_block_store.py` — including `reorder_batch_rows`'s reorder/duplicate/drop behavior across both tiers, the primitive beam search reorders on — and `test_streaming_attention.py`, the Triton kernels/batching/padding-mask correctness, plus Gemma2's custom scale/softcapping against a dense reference and a regression guard that the `scale=None`/`softcap=None` defaults are unchanged from before those parameters existed) and are marked `skipif(not torch.cuda.is_available())`, so they run for real locally but are silently skipped on CI's GPU-less runners rather than failing.
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
- Everything above was validated on one weak, old GPU (GTX 1050 Ti, Pascal, 4GB) and models under 2B params. First real-scale test, on a rented NVIDIA A40 (Ampere, 46GB) with `Qwen/Qwen2.5-7B-Instruct` (28 layers, `head_dim=128` — 2x every model tested before), 7,875-token context (`bench/streaming_paged_model_scale_test_mvp.py`), surfaced two things worth knowing before trying this on a bigger model yourself:
  - **`float16` overflows to `NaN` in the logits by the second decode step at this model size and context length, with the *unmodified, unpaged* baseline** — nothing to do with `pager_hf`, a well-known fp16 dynamic-range limitation that shows up earlier on bigger models at longer contexts. Once logits are `NaN`, `argmax` is undefined, so any two independently-computed generations can legitimately land on different tokens without either being "wrong" — this is exactly what a first pass at this test looked like (`streaming_ids` differing from `baseline_ids`), before tracing it back to the baseline's own `NaN`, not a pager bug. Switching to `bfloat16` (much wider dynamic range, standard practice for models this size) removed the `NaN` entirely and made `baseline`/non-streaming/streaming agree on every token again, exactly as at every smaller scale tested so far. **Use `bfloat16`, not `float16`, once models get into this range** — `pager_hf` itself is dtype-agnostic and does whatever the caller loads the model as.
  - **The streaming kernel's speed advantage did not hold at this scale at first — it reversed, then got fixed.** At 0.5B, streaming was ~2.1x baseline throughput. At 7B, streaming initially ran at 0.72-0.87x of the non-streaming path's throughput instead — *slower*, depending on policy. A first attempt at a fix — auto-tuning `block_kv`/`num_warps` per `head_dim` on top of the existing kernel design — was tried and reverted: it didn't corrupt anything (a first, unsafe `@triton.autotune`-based version did, and was caught by the test suite before shipping; a safe manual-tuning replacement wasn't unsafe, just picked a config that made things measurably *worse*, 0.58x). That whole detour turned out to be optimizing the wrong layer. Two real root causes, found by profiling the actual pipeline rather than guessing further:
    1. **The kernel's own math never touched tensor cores, on any hardware.** The original design computed QK^T/P·V as broadcast-multiply + `tl.sum` (one program per query head) — correct, but plain FMA on any GPU, Ampere included. PyTorch's dense "eager" attention path uses `torch.matmul`, real cuBLAS GEMM, which *does* get tensor-core acceleration on Ampere — an advantage the dev GPU (GTX 1050 Ti, Pascal, no tensor cores at all) could never reveal, which is exactly why this shipped without anyone noticing at dev scale. Fixed by packing the `n_rep` query heads that share a KV head into one `tl.dot`-eligible `[16, head_dim]` tile per program (grid shrinks from `batch*num_query_heads` to `batch*num_kv_heads`, padded rows masked out) — tensor-core-eligible on Ampere+, plain FMA on Pascal, correct and tested on both. A tiny edge case fell out of this: `tl.dot` needs its contraction dimension ≥16, which broke on a `head_dim=8` toy test checkpoint (`tiny-random-MistralForCausalLM`) — fixed by zero-padding Q/K up to 16 columns for that dot specifically (mathematically a no-op; real models are always `head_dim≥64` so this never triggers for anything that matters).
    2. **The bigger one: a per-block Python gather loop, not the attention math, was the actual bottleneck.** Profiling one real decode step (`bench/streaming_gather_vs_kernel_profile_mvp.py`, instrumenting the real `PagedModel`/`KVBlockStore` path directly) found `_gather_layer_kv_group` — which loops over every block in a group individually, doing a `.to(device)` plus a separate slice-assignment into a shared destination tensor per block — eating **64% of total decode time**, versus **15%** for the attention kernel itself. The kernel fix above was real and correct, but it was optimizing a piece that was never the dominant cost at this scale; the same "many tiny ops pay kernel-launch overhead" pattern that made this project's very first pure-Python attention prototype impractical had resurfaced in the KV-gather path instead. Fixed by collecting each block's (permuted, still-a-view) K/V slice into a Python list and combining the whole group with one `torch.cat` instead of N separate slice-assignments — same data moved, far fewer kernel launches.
    
    Both fixes verified together on the real A40/7B/8000-token setup: `same_token_ids` still holds exactly against the unpaged baseline, peak-memory reduction unchanged (10.7%), and **streaming is now faster than non-streaming at this scale** — 1.05x for `recent_only`, 1.17x for `sinks_heavy_hitter` (the heavier two-pass, mass-tracking path). The regression is closed, not just explained.
- Architecture coverage extended beyond Qwen2/Llama/Mistral: **Qwen2MoE, Starcoder2** (both use the same "qwen2-style" RoPE convention and `self_attn` shape as `Qwen2Attention` — MoE only changes the MLP; their sliding-window config defaults off, and this transformers version's eager-mode causal-mask builder doesn't enforce one anyway when it is on, so there's nothing to replicate) and **Gemma** (llama-style RoPE, no gotchas in this transformers version) — verified `same_token_ids` for all 4 policies, streaming and non-streaming, on real (if tiny, randomly-initialized) checkpoints of each (`bench/streaming_paged_model_verify_new_archs_mvp.py`).
- **Phi3 and Gemma2** are supported too now, closing the two remaining "deliberately not added" entries from the previous update that were tractable: Phi3 fuses `q_proj`/`k_proj`/`v_proj` into one `qkv_proj` linear (split the same way `Phi3Attention.forward` does, see `_project_qkv` in `paged_model.py`) but otherwise turned out to need *zero* new RoPE code — `Phi3RotaryEmbedding`/its `apply_rotary_pos_emb` are byte-identical to Llama's own, confirmed by diffing the actual source. Gemma2 needed real Triton kernel changes: a non-default QK^T scale (`query_pre_attn_scalar`-based, not `1/sqrt(head_dim)`) and attn-logit softcapping (`tanh(scores/cap)*cap`, applied to raw scores before masking) — both threaded through all four kernels as new `scale`/`softcap` arguments (`None`/`0.0` defaults preserve every other architecture's exact prior behavior), plus **real sliding-window attention on alternating layers**, which turned into the bigger finding of this pass (see below). Phi (not Phi3, `o_proj` named `dense`), StableLm (partial rotary), Olmo/Cohere (QK clipping/QK-norm) remain out of scope for the same reasons as before. Unrecognized architectures still raise `NotImplementedError` rather than guessing.
- **Real sliding-window truncation, and a correction to what this README used to claim about it.** Investigating Gemma2 found that its alternating-layer sliding window was not just missing for Gemma2 — the streaming path's patched attention forward accepts an `attention_mask` argument but never reads it, building its own mask purely from left-padding info with no window bound at all, for *any* architecture. This README previously said sliding window was "already baked into the attention_mask the same way Mistral's is" for the new architectures added earlier — true for the **non-streaming** reload path (real, unpatched HF forward), but not for streaming, and Mistral's shipped, already-verified support had silently had this exact gap the whole time (its real eager-mode baseline *does* enforce a uniform `sliding_window=4096` via `MistralModel._update_causal_mask`, confirmed by reading its source — it just never mattered because every test/bench prompt here is far under 4096 tokens). Fixed by reusing the existing `valid`-mask exclusion mechanism (no new kernel capability needed): a per-layer window bound (`paged_model._sliding_window_for_layer` — read directly from `Gemma2Attention`'s own already-computed `self.sliding_window` for Gemma2's alternating layers, from `config.sliding_window` uniformly for Mistral) excludes any position further back than the window, folded into the same mask every group/tail chunk already derives from. Verified on tiny, randomly-initialized Mistral and Gemma2 configs with a deliberately small `sliding_window` (24) and a prompt genuinely longer than it (`bench/streaming_paged_model_verify_sliding_window_mvp.py`) — since every other prompt in this project is far shorter than any real checkpoint's window, this is the only test that actually exercises the truncation rather than it being a no-op. **A second, real-A40 verification pass found the test's own "ground truth" was wrong for Gemma2**, not this project's code: `transformers==4.44.2`'s `Gemma2DecoderLayer.forward` builds its sliding-window exclusion as `torch.tril(torch.ones_like(attention_mask), diagonal=-sliding_window)`, which indexes by the *local* row position within the query dimension, not the absolute sequence position — during prefill (`q_len == kv_len`) those coincide and the window is enforced correctly, but during ordinary single-token incremental decode (`q_len == 1`) the only row index is 0, so the exclusion condition can never hold for a positive window and the tril mask is silently all-`False`: confirmed directly by hooking `Gemma2Attention.forward` and inspecting the real `attention_mask` tensor on a decode step (no `-inf` anywhere, even far outside the window). The bench script's hand-rolled baseline called the model the same one-token-at-a-time way, so it inherited this exact HF bug and never actually enforced a window past the first prefill chunk — the small, "near-tie" divergence seen in an earlier run of this same test was this same reference bug, not floating-point noise; different random weights (from a different `torch` version's RNG) just made it large enough to stop looking like one. Fixed the *test*, not the product code: the baseline now recomputes the whole sequence-so-far from scratch at every step (fresh cache, `q_len == kv_len` on every call), which sidesteps the HF bug entirely. Against that corrected reference, streaming now matches **exactly** for both Mistral and Gemma2, no near-tie exception needed. **Scoped to the single-session path only**: `batched_decode.py`'s cross-session `batched_decode_step` batches rows that can be at different absolute positions, so the window bound would need to be per-row there, not a single scalar — not attempted in this pass, so `batched_decode_step` now explicitly rejects Mistral/Gemma2 with a clear error rather than silently computing an unwindowed (wrong) result for them; Stage 1 round-robin serving (`generate()` per session) is unaffected.
- **Caught and fixed a real, pre-existing, generic bug along the way, unrelated to any single architecture:** `PagedModel._forward_step`'s non-streaming (reload) branch and `_chunked_prefill`'s multi-chunk loop never passed `cache_position` to the model, silently defaulting to HF's own `torch.arange(0, chunk_len)` (i.e. always restarting from 0, ignoring how much context is already cached). This happened to be harmless for every previously-tested architecture (their causal-mask builders don't depend on `cache_position`'s absolute value for a plain `DynamicCache`) but produced a genuinely wrong attention mask for Gemma2, whose `_update_causal_mask` uses it directly — found via real-model verification (a large, easy-to-spot divergence), not by code inspection. Fixed by passing the correct absolute `cache_position` explicitly in both places.
- `generate_beam_search` gained the follow-ups originally deferred when it shipped: a batch of independent prompts (`batch_size > 1`, each running its own `num_beams`-wide search, never mixing candidates across prompts), `length_penalty` (default `1.0`, matching HuggingFace's own default; only `0.0` is verified byte-exact against HuggingFace's reference, nonzero values aren't guaranteed bit-for-bit identical to HuggingFace's own separate hypothesis-bookkeeping), `num_return_sequences`, and `do_sample` as a "stochastic beam search" hybrid (sampling without replacement from the filtered joint candidate distribution each step, checked only for valid non-crashing output — the two implementations' RNGs don't match, so exact agreement isn't the bar there). All four verified against HuggingFace's real reference where an exact bar exists — `length_penalty=0.0` and `1.0`, `num_return_sequences=2`, and `batch_size=2` all match exactly (`bench/real_kv_beam_search_mvp.py`).

---

## License

Apache-2.0. See [LICENSE](LICENSE).

---

## Project stage

Research prototype, not production software.

Current milestone: a real, installable API (`pager_hf.PagedModel`) that pages KV cache for any of the four policies, at both short and long context, with session persistence, batched generation, temperature/top-k/top-p sampling, an enforced `ram_budget`, and safe concurrent-call rejection — all checked byte-identical against an unpaged baseline (for the greedy default), plus a Rust/Python test suite in CI.

The peak-memory ceiling that used to sit here as the "real next problem" (see [Limitations](#important-limitations)) is now solved in the shipped product, not just `bench/`, and on by default: a real custom Triton kernel (`pager_hf/streaming_attention.py`), online-softmax streaming attention that carries its accumulator state across separate kernel launches, gathering only a small group of blocks into GPU memory at a time instead of the whole context. All four policies are supported, including the two that score blocks by attention (`heavy_hitter`, `sinks_heavy_hitter`) via a second Triton pass that computes per-block attention mass directly, without needing HuggingFace's own `output_attentions=True`. Measured on `Qwen2.5-0.5B-Instruct`, on the integrated `PagedModel` itself (`bench/streaming_paged_model_peak_memory_mvp.py`), with a constrained `vram_budget` so blocks genuinely offload between steps: 8.5% peak-memory reduction during the decode step at 3,000 tokens, 16.2% at 5,922 tokens — growing with context length, as it should — with `same_token_ids: True` against both an unpaged baseline and the reconstruct-and-call path, across all four policies (`bench/streaming_paged_model_verify_mvp.py`). The same kernel is also just fast enough to flip the project's own throughput number: one layer's attention over the full context runs about 10x faster than the entire current 24-layer step, and with the old per-step reload gone too, paged decode went from ~1.17x *slower* than baseline to ~2.1x *faster* (see [Limitations](#important-limitations)). vLLM was considered as a shortcut (reuse their kernel, add this project's Rust-scored placement on top) and ruled out on this exact hardware: it hard-requires compute capability ≥7.5, and the GTX 1050 Ti (6.1) simply isn't in range — confirmed by a closed upstream PR, not just documentation.

What used to be opt-in caveats are now closed too: `use_streaming_attention` (now the default) supports `batch_size > 1` for *every* policy, including `heavy_hitter`/`sinks_heavy_hitter`, and Qwen2-, Llama-, and Mistral-family decoder layers — verified on `TinyLlama-1.1B-Chat` and a real Mistral checkpoint in addition to `Qwen2.5-0.5B-Instruct` (see [Limitations](#important-limitations) for what actually differs between the RoPE conventions under the hood). The group size that bounds the gather buffer was swept (`bench/streaming_group_size_sweep_mvp.py`) and bumped from an untuned 16 to 64. Structured logging was also added (`logging.getLogger("pager_hf")`) so a long-lived process has some visibility into session lifecycle, per-step transfers, and `ram_budget` pressure without needing its own instrumentation.

First real-scale test off the dev GTX 1050 Ti: a rented NVIDIA A40 running `Qwen2.5-7B-Instruct` at ~8,000 tokens (`bench/streaming_paged_model_scale_test_mvp.py`) confirmed correctness and the peak-memory reduction both hold at a real model size and context length that were never possible to test on 4GB before (10.7%, in the same range as at 0.5B) — but also found the streaming kernel's throughput advantage did *not* generalize to this scale; it reversed to 0.72-0.87x of the non-streaming path instead of the ~2.1x seen at 0.5B. Root-caused to two combined issues — the kernel never used tensor cores, and a per-block Python gather loop was actually the dominant cost (64% of decode time, not the attention math's 15%) — both fixed and re-verified on the same A40: streaming is now faster than non-streaming at 7B scale too (1.05x-1.17x depending on policy). See [Limitations](#important-limitations) for the full account, including a real fp16-`NaN`-at-scale finding along the way that turned out to be unrelated to this project entirely.

First serving-framework integration: `pager_hf.serving` (optional `pip install pager-hf[serve]`), a FastAPI server that lets more than one request run without each one blocking the next until it fully finishes — see [Serving multiple concurrent requests](#serving-multiple-concurrent-requests). Deliberately scoped as **Stage 1**: a round-robin scheduler interleaves independent `PagedModel` sessions one decode step at a time (verified token-exact and genuinely interleaved on a real model, `bench/serving_concurrent_requests_mvp.py`), not vLLM-style batched continuous batching (multiple sessions combined into one GPU forward call) — that's structurally blocked by the current design (`use_streaming_attention` monkey-patches the *shared* model's decoder layers, so only one `generate()` call is ever safely in flight) and would need real surgery to add, tracked as a separate Stage 2, not attempted here.

While building the serving layer, found and fixed a real pre-existing bug, unrelated to serving itself: any prompt shorter than one `tokens_per_block` (16 tokens by default) left `_num_blocks` at 0 for the first several decode steps (everything still in the tail, nothing promoted to a full pager block yet) — and every policy hit it, just with a different failure. `recent_only`/`sinks_recent` raised `ZeroDivisionError` (a uniform `1/num_blocks` placement vector); `heavy_hitter`/`sinks_heavy_hitter` raised `OverflowError` (`query_block = num_blocks - 1` went negative, rejected by the Rust pager's `u64` argument); the non-streaming reload path additionally raised `KeyError: 'Block 0 is not on GPU.'` from `reconstruct_past_from_store`, which assumed block 0 always existed just to sample a tensor shape/dtype from it. Fixed in `PagedModel._forward_step` (skip the pager's score/pin/rebalance step entirely when there's nothing to place yet) and `reconstruct_past_from_store` (derive shape/dtype from the tail instead of an assumed block 0). Verified: all 4 policies × streaming and non-streaming (8 combinations) now match an unpaged baseline exactly on a 15-token prompt (`bench/real_kv_paged_model_short_prompt_mvp.py`), plus two new CPU-only unit tests (`tests/test_kv_utils.py`).

Beam search (`generate_beam_search`) is shipped too — see [Beam search](#beam-search). Reordering a beam's whole KV-cache history into a new row (a strong beam spawning more than one child, a weak one dying) turned out to be a single `KVBlockStore.reorder_batch_rows` call, since blocks already keep batch as a plain tensor dimension; the actual work was getting the beam-search bookkeeping itself right, including a real first-step bug (every beam starts identical, so naive score initialization picks the same top token from duplicate rows instead of the true top-*k* distinct tokens). Verified against HuggingFace's own `model.generate(num_beams=...)` reference, not just internal consistency — exact match, streaming and non-streaming.

The three items flagged as remaining right after the serving layer shipped are now all closed too: `use_streaming_attention` covers **Qwen2MoE, Starcoder2, and Gemma** in addition to Qwen2/Llama/Mistral (see [Important limitations](#important-limitations) for which architectures were deliberately left out and why — Phi/Phi3/StableLm/Gemma2/Olmo/Cohere each need real per-architecture handling, not just a `model_type` string); `generate_beam_search` gained a batch of independent prompts, `length_penalty`, `num_return_sequences`, and a `do_sample` hybrid, all verified against HuggingFace's reference where an exact bar exists; and **Stage 2's batched cross-session decode is now real, not just a proven-but-unwired mechanism.** `pager_hf.batched_decode.batched_decode_step` (see [Batched cross-session decode](#batched-cross-session-decode)) batches several independent sessions' decode steps into one `model.forward()` call, verified correct against isolated `generate()` runs and measurably faster than pure round-robin (1.25-1.3x on the dev GPU) — caught and fixed a real `NaN` edge case in the shared Triton kernel along the way (a session with zero real history batched alongside longer ones hit an indeterminate `-inf - -inf` in the online-softmax update). That mechanism is now the default path `pager_hf.serving`'s scheduler actually takes: every round it groups already-started sessions by tail length and advances each group with one combined forward call, falling back to a plain per-session `generate()` call only for a fresh session's first step or a session with no batching partner this round — verified end to end (`bench/serving_concurrent_requests_mvp.py`) with the scheduler's own batched-step method instrumented directly, not just trusted by design: 3 real concurrent same-prompt requests all match the isolated reference exactly, and the batched path measurably fires for them. Then: publish `rust-llm-pager-core` and `pager-hf` to PyPI — packaging is done and tested locally, just not uploaded yet.

Three more honest production-readiness gaps closed since then. **Real scale**, not just 3 toy sessions on a 0.5B model on a weak dev GPU: re-verified `batched_decode_step`/the serving scheduler with 8 concurrent sessions on real `Qwen2.5-7B-Instruct` (`bfloat16`) on a rented A40 (`bench/real_kv_batched_decode_scale_mvp.py`) — 2.9-3.0x speedup vs. round-robin, 7 of 8 sessions matching their isolated reference byte-for-byte, the 8th diverging only at a genuine, reproducible bf16 near-tie (see [Batched cross-session decode](#batched-cross-session-decode)). **Padding-to-max no longer means a short session pays full compute for a long one's padding**: every kernel program in `batched_decode_step`'s path is already one-per-row, so a per-row `active` flag now lets a row with zero real data in a given KV group skip that group's loop entirely instead of running it and having every position masked out anyway — a short session batched next to several long ones now costs a fraction of what it used to (measured ~0.34x of a long session's own cost on the dev GPU for a 1-block-vs-24-block skew, `bench/real_kv_batched_decode_skew_mvp.py`), not full vLLM-parity per-row block tables (that still needs `KVBlockStore`'s storage model to change, not attempted). **Multi-turn conversations** — see [Serving multiple concurrent requests](#serving-multiple-concurrent-requests): `pager_hf.serving` gained `session_id` support (`POST /v1/sessions`, `DELETE /v1/sessions/{id}`), exposing `PagedModel.generate()`'s own already-existing session persistence over HTTP so a client's KV cache survives across separate `/v1/completions` calls instead of every call starting over — verified end to end against a direct `PagedModel` session generating the same two turns.

Two more gaps closed after that, finishing the honest production-readiness list. **Operational tooling**: `/healthz`, `/metrics` (Prometheus), opt-in `--api-key` auth, a real draining `stop(timeout=...)` wired to SIGTERM/SIGINT for graceful shutdown, and `--max-queue-depth` backpressure (HTTP 429 + `Retry-After`) — see the "Operational tooling" part of [Serving multiple concurrent requests](#serving-multiple-concurrent-requests). Graceful shutdown was verified on a real running server, not just unit tests: a 40-token request in flight when SIGTERM arrived completed in full before the process exited. **Multiple GPUs, one process**: `--devices cuda:0,cuda:1,...` runs one full model replica plus its own scheduler per GPU, load-balanced by a new `MultiGpuScheduler` router that duck-types the same interface a single scheduler exposes — data-parallel replicas, not model/tensor parallelism, and single-machine only (cross-machine load balancing is a reverse proxy's job, using the `/healthz`/`/metrics` endpoints already built). The routing/session-affinity logic is unit-tested against fakes; genuine multi-GPU hardware verification is still pending (needs a rented 2+ GPU instance, unavailable at the time this was built), flagged explicitly rather than assumed to work.

Only PyPI publish (still user-deferred) and each feature's own disclosed first-pass gaps (no idle-timeout session eviction, independent not joint session/round-robin capacity accounting, single shared-secret auth, and the still-pending multi-GPU hardware re-verification) remain.

**Last architecture-coverage gap closed: Phi3 and Gemma2.** See [Important limitations](#important-limitations) for the technical account — Phi3 needed only a fused-`qkv_proj` split (its RoPE is byte-identical to Llama's own); Gemma2 needed real Triton kernel changes (a non-default QK^T scale, attn-logit softcapping) plus real sliding-window truncation on its alternating layers, which turned into the bigger finding here: the streaming path never enforced *any* sliding window, a silent gap in the already-shipped Mistral support too (fixed for both, single-session path; `batched_decode_step` explicitly rejects both architectures for now rather than risk an unwindowed wrong result there). Along the way, a real, generic, pre-existing bug surfaced and got fixed: `PagedModel`'s non-streaming reload path and multi-chunk prefill never passed `cache_position` to the model, which happened to be harmless for every architecture verified so far but produced a genuinely wrong attention mask for Gemma2 specifically. This investigation is also a good example of the discipline this project tries to hold itself to under pressure to just ship something: the first correctness run showed streaming diverging from a hand-rolled "baseline" script, and it would have been easy to conclude Gemma2 support was broken — instead, tracing the *reference* itself down to a missing `cache_position` (not the new kernel code) found that streaming had been correct all along, and surfaced a real bug (affecting Mistral too) that a shallower check would have missed entirely.

Re-verified this whole day's work end to end on a rented A40 (the dev machine has no GPU at all): full test suite (92/92), `black`/`isort` clean, and every real-GPU bench script — architecture coverage (`bench/streaming_paged_model_verify_new_archs_mvp.py`), the padding-skip optimization (`bench/real_kv_batched_decode_skew_mvp.py`), multi-turn sessions/operational tooling/Stage 2 batching against real HTTP traffic (`bench/serving_concurrent_requests_mvp.py`), and the two pre-existing regression benches (short-prompt, beam search) — all passed. One real finding along the way, in the *test* rather than the product: the dedicated sliding-window bench (`bench/streaming_paged_model_verify_sliding_window_mvp.py`) initially failed for Gemma2 on this hardware, with all four policies diverging identically and well outside near-tie range — different random weights than whatever run had previously accepted a 0.0017-gap near-tie made a *pre-existing* bug in that bench's own hand-rolled reference impossible to ignore instead of easy to wave off. Root-caused (confirmed by hooking `Gemma2Attention.forward` and inspecting the real `attention_mask` tensor on a decode step) to `transformers==4.44.2`'s own `Gemma2DecoderLayer.forward`: its sliding-window exclusion (`torch.tril(..., diagonal=-sliding_window)`) is built from the query dimension's *local* row index, which only coincides with the absolute sequence position during prefill (`q_len == kv_len`) — during ordinary single-token decode the only row index is 0, so the exclusion can never trigger and the window silently goes unenforced past the first prefill chunk. The bench's own baseline decoded one token at a time the same way HF's `generate()` does, so it inherited this exact bug. Fixed the test (not the product): the baseline now recomputes the whole sequence from scratch every step (fresh cache, always `q_len == kv_len`), sidestepping the bug entirely — against that corrected reference, streaming now matches Gemma2 and Mistral **exactly**, no near-tie exception needed. Multi-GPU hardware verification is still the one item nobody has been able to test for real: this rented instance has one A40, not two, so `MultiGpuScheduler`'s routing logic remains proven only against fakes.
