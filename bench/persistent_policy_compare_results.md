# Persistent CPU KV paging: policy comparison

Model: `Qwen/Qwen2.5-0.5B-Instruct`  
Generated tokens per policy: 64

| Policy | Same as baseline | New blocks | Final blocks | Mean attn in VRAM | Min attn in VRAM | GPU->CPU MB | CPU->GPU MB |
|---|---|---:|---:|---:|---:|---:|---:|
| `heavy_hitter` | True | 4 | 13 | 0.8717 | 0.5137 | 49.55 | 48.37 |
| `sinks_heavy_hitter` | True | 4 | 13 | 0.8627 | 0.6805 | 49.55 | 48.37 |
| `recent_only` | True | 4 | 13 | 0.6244 | 0.3686 | 49.55 | 48.37 |
| `sinks_recent` | True | 4 | 13 | 0.6244 | 0.3686 | 49.55 | 48.37 |
