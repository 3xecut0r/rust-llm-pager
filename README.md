# rust-llm-pager

A Rust/PyO3 prototype for KV-cache paging in LLM inference.

`rust-llm-pager` experiments with moving KV-cache blocks between GPU and CPU memory while preserving generation correctness. The project starts with a Rust policy simulator, then progressively validates the policy on real PyTorch tensors and real `past_key_values` from `Qwen/Qwen2.5-0.5B-Instruct`.

The current prototype is not a production inference engine yet. It is a research/engineering milestone that proves a Rust pager can control KV-cache block placement and preserve exact greedy generation after GPU ↔ CPU movement.

---

## Why this exists

LLM inference becomes memory-bound quickly as context length grows. The KV cache can consume a large amount of VRAM, especially for long prompts, concurrent requests, or smaller GPUs.

The goal of this project is to explore whether a small Rust pager can decide which KV blocks should stay in VRAM and which blocks can be moved to slower tiers such as CPU RAM or eventually SSD.

The long-term goal is to integrate this idea with runtimes such as vLLM, LMCache, or another inference backend.

---

## Current status

Implemented and validated:

- Rust pager core with PyO3 bindings.
- Multiple KV placement policies.
- Multi-needle soft-masked benchmark.
- Real PyTorch KV-like GPU ↔ CPU movement.
- Real Qwen `past_key_values` extraction and block splitting.
- Exact KV block roundtrip validation.
- Identical next-token logits after KV reconstruction.
- Identical multi-token greedy generation after KV reconstruction.
- Paged real Qwen generation loop.
- Baseline vs paged generation comparison with identical token IDs.

Current headline result:

```text
OK: paged real Qwen KV loop produces identical greedy generation to baseline.
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

The Rust side decides block placement. The Python side uses PyTorch to store and physically move tensors.

---

## Policies

Implemented policies:

| Policy | Description |
|---|---|
| `recent_only` | Keeps the most recent blocks in VRAM. |
| `sinks_recent` | Keeps sink blocks plus recent blocks. |
| `heavy_hitter` | Keeps blocks with high accumulated attention score. |
| `sinks_heavy_hitter` | Hybrid policy: sink blocks + limited recent tail + high-attention blocks. |

The strongest current policy is:

```text
sinks_heavy_hitter
```

It keeps:

1. early sink blocks,
2. a limited recent tail,
3. the remaining VRAM budget filled with high-attention blocks.

---

## Benchmark result

The multi-needle soft-masked benchmark evaluates 7 needle-in-haystack prompts across 15 penalty settings per prompt.

Main result:

| Policy | Wins |
|---|---:|
| `sinks_heavy_hitter` | 86 / 105 |
| `recent_only` | 9 / 105 |
| `heavy_hitter` | 8 / 105 |
| `sinks_recent` | 2 / 105 |

For strong SSD penalties (`SSD <= -6`):

| Policy | Wins |
|---|---:|
| `sinks_heavy_hitter` | 55 / 56 |
| `sinks_recent` | 1 / 56 |
| `heavy_hitter` | 0 / 56 |
| `recent_only` | 0 / 56 |

Aggregate result:

| Policy | Mean PPL | Median PPL | Attention in VRAM | Swap total |
|---|---:|---:|---:|---:|
| `sinks_heavy_hitter` | ~384.3 | ~18.5 | ~0.660 | ~3.36 GB |
| `heavy_hitter` | ~2866.9 | ~420.7 | ~0.543 | ~3.05 GB |
| `sinks_recent` | ~3158.5 | ~1221.9 | ~0.386 | ~2.18 GB |
| `recent_only` | ~4414.3 | ~1149.4 | ~0.014 | ~2.18 GB |

The hybrid policy fixes failure modes where pure heavy-hitter misses important recent or sink context.

---

## Real tensor movement MVP

The project includes a first real tensor movement prototype outside of vLLM.

This MVP creates KV-like PyTorch tensors, stores them as blocks, and lets the Rust pager decide which blocks should stay on GPU. All non-VRAM tiers are currently mapped to CPU.

Current scope:

- Real CUDA tensors.
- Real GPU → CPU copies.
- Real CPU → GPU copies.
- Rust pager controls physical tensor placement.
- SSD tier is currently mapped to CPU / simulated.
- Not yet integrated into a production inference runtime.

Example result:

| Metric | Value |
|---|---:|
| Total blocks | 160 |
| GPU-resident blocks | 15 |
| CPU-resident blocks | 145 |
| Real GPU resident KV | ~11.8 MB |
| Real CPU resident KV | ~114.0 MB |
| GPU → CPU copied | ~126.6 MB |
| CPU → GPU copied | ~12.6 MB |
| Attention in VRAM | ~49.6% |

The pager-selected VRAM blocks match the actual GPU-resident tensor blocks in the store:

```text
pager: [0, 1, 2, 3, 20, 21, 55, 56, 146, 147, 148, 149, 150, 151, 152]
store: [0, 1, 2, 3, 20, 21, 55, 56, 146, 147, 148, 149, 150, 151, 152]
```

This proves that the Rust policy can control real KV-like tensor placement.

---

## Real Qwen KV-cache block MVP

The project extracts real `past_key_values` from `Qwen/Qwen2.5-0.5B-Instruct`, splits them into KV blocks, and applies Rust pager placement to those real model KV tensors.

Example run:

| Metric | Value |
|---|---:|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Layers | 24 |
| KV heads | 2 |
| Head dim | 64 |
| Prompt sequence length | 151 |
| Tokens per block | 16 |
| Full KV blocks | 9 |
| GPU-resident blocks | 7 |
| CPU-resident blocks | 2 |
| Real model KV block size | ~0.197 MB |
| Real GPU-resident KV | ~1.38 MB |
| Real CPU-resident KV | ~0.39 MB |
| Real attention kept on GPU | ~94.0% |

The pager-selected VRAM blocks match the actual GPU-resident KV blocks:

```text
pager: [0, 1, 2, 3, 6, 7, 8]
store: [0, 1, 2, 3, 6, 7, 8]
```

This proves that the Rust pager can control placement of real model `past_key_values`, not only dummy KV-like tensors.

---

## KV roundtrip correctness

The project validates that real Qwen KV blocks survive GPU → CPU → GPU movement exactly.

Test flow:

```text
real Qwen past_key_values
-> split into KV blocks
-> move selected blocks GPU -> CPU
-> reload blocks CPU -> GPU
-> reconstruct past_key_values
-> compare with original KV tensors
```

Example result:

```text
Roundtrip comparison
--------------------
max_key_diff: 0.0
max_value_diff: 0.0

OK: real Qwen KV blocks survived GPU -> CPU -> GPU roundtrip exactly.
```

---

## Logits correctness

The project validates that reconstructed real Qwen KV cache produces identical next-token logits.

Test flow:

```text
original past_key_values -> next-token logits
reconstructed past_key_values -> next-token logits
```

Example result:

```text
Logits roundtrip comparison
---------------------------
max_logits_diff: 0.0
mean_logits_diff: 0.0
same_argmax: True
original_argmax: 304 ' in'
reconstructed_argmax: 304 ' in'

OK: reconstructed real Qwen KV cache produces identical next-token logits.
```

---

## Multi-token generation roundtrip

The project validates that reconstructed real Qwen KV cache can produce identical multi-token greedy generation.

Test flow:

```text
Qwen past_key_values
-> split into KV blocks
-> GPU -> CPU offload
-> CPU -> GPU reload
-> reconstruct past_key_values
-> greedy decode from reconstructed cache
```

Example result:

| Metric | Value |
|---|---:|
| Generated tokens | 8 |
| Same token IDs | `true` |
| Max KV diff after roundtrip | `0.0` |
| Max logits diff after roundtrip | `0.0` |

Example generated text:

```text
original:      ' in the first paragraph? The secret project'
reconstructed: ' in the first paragraph? The secret project'
```

This confirms that reconstructed real Qwen KV cache can produce identical multi-token greedy generation.

---

## Paged generation loop

The project includes a multi-step paged generation loop prototype.

The loop does the following on every generation step:

```text
real past_key_values
-> split full KV blocks
-> keep incomplete tail tokens hot
-> Rust pager chooses GPU/CPU placement
-> offload cold full blocks GPU -> CPU
-> reload full blocks for HuggingFace forward
-> reconstruct full-block KV
-> append tail KV back
-> generate next token
-> repeat
```

This is still not an efficient production runtime because HuggingFace forward currently receives a full reconstructed GPU cache. However, it demonstrates that the pager can operate inside a real generation loop without losing correctness.

Example result:

```text
generated_text: ' BLUE ORCHID. The secret project'
total_gpu_to_cpu_mb: 3.15
total_cpu_to_gpu_mb: 3.15
total_gpu_to_cpu_copies: 16
total_cpu_to_gpu_copies: 16

OK: real Qwen KV cache was paged in a multi-step generation loop.
```

---

## Baseline vs paged Qwen generation

The strongest current correctness demo compares normal Qwen greedy generation against the paged KV loop.

Test flow:

```text
baseline:
Qwen cache -> greedy decode

paged:
Qwen cache
-> split full KV blocks
-> keep tail tokens hot
-> Rust pager chooses GPU/CPU placement
-> offload cold blocks GPU -> CPU
-> reload required blocks CPU -> GPU
-> reconstruct cache
-> append tail KV
-> greedy decode
```

Example result:

| Metric | Value |
|---|---:|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Generated tokens | 8 |
| Same token IDs as baseline | `true` |
| Total GPU → CPU copied | ~3.15 MB |
| Total CPU → GPU copied | ~3.15 MB |
| GPU → CPU copies | 16 |
| CPU → GPU copies | 16 |
| Mean attention kept on GPU | ~93.6% |
| Min attention kept on GPU | ~91.5% |
| Max attention kept on GPU | ~95.4% |

Example output:

```text
baseline_ids: [55892, 2726, 2149, 915, 13, 576, 6234, 2390]
paged_ids:    [55892, 2726, 2149, 915, 13, 576, 6234, 2390]
same_token_ids: True
baseline_text: ' BLUE ORCHID. The secret project'
paged_text:    ' BLUE ORCHID. The secret project'

OK: paged real Qwen KV loop produces identical greedy generation to baseline.
```

This confirms that the paged real Qwen KV loop can preserve exact greedy generation while moving KV blocks between GPU and CPU.

---

## Repository layout

```text
pager/
  Cargo.toml
  Cargo.lock
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
```

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

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -U pip maturin
pip install torch transformers accelerate
```

If using a CUDA-specific PyTorch wheel, install PyTorch according to your CUDA version first.

Example used during development:

```text
torch==2.3.1+cu118
transformers==4.44.2
```

---

## Build

Build the Rust/PyO3 extension:

```bash
maturin develop -m pager/Cargo.toml --release
```

Then verify that Python can import the module:

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

## Run demos

Compile-check Python files:

```bash
python -m py_compile bench/*.py
```

Run real tensor movement:

```bash
python bench/torch_kv_offload_mvp.py
python bench/torch_kv_pager_integration_mvp.py
```

Run real Qwen KV demos:

```bash
python bench/real_kv_cache_block_mvp.py
python bench/real_kv_roundtrip_mvp.py
python bench/real_kv_logits_roundtrip_mvp.py
python bench/real_kv_generation_roundtrip_mvp.py
python bench/real_kv_paged_generation_loop_mvp.py
python bench/real_kv_paged_generation_compare_mvp.py
```

The most important current demo is:

```bash
python bench/real_kv_paged_generation_compare_mvp.py
```

Expected final line:

```text
OK: paged real Qwen KV loop produces identical greedy generation to baseline.
```

---

## Important limitations

This is a prototype, not a production inference backend.

Current limitations:

- Not integrated with vLLM yet.
- Not integrated with LMCache yet.
- HuggingFace forward still receives a reconstructed full GPU cache.
- CPU/SSD tiering is not yet optimized.
- SSD tier is simulated or mapped to CPU in current MVPs.
- Benchmarks are small and designed for correctness/proof-of-concept, not final performance claims.
- The Rust pager uses a logical block size for placement metrics; PyTorch demos also report real tensor bytes separately.
- Current real model demos use `Qwen/Qwen2.5-0.5B-Instruct` only.

What this project currently proves:

- Rust can control KV block placement.
- Real tensors can be moved GPU ↔ CPU.
- Real Qwen KV blocks can survive offload/reload exactly.
- Reconstructed real Qwen KV can produce identical logits.
- The paged loop can produce identical greedy generation to baseline.

What it does not yet prove:

- End-to-end throughput improvement.
- Lower latency in a production inference engine.
- Native runtime integration with paged attention kernels.
- SSD-backed production offload.

---

## License
 
### Apache-2.0

---

## Project stage

Current stage:

```text
research prototype / MVP
```

Current milestone:

```text
baseline-equivalent paged Qwen KV generation with real GPU <-> CPU tensor movement
```

Next milestone:

```text
turn the MVP into a clean open-source repo and start runtime integration research
```