import pandas as pd

RESULTS_PATH = "bench/multi_needle_soft_masked_grid_results.csv"

df = pd.read_csv(RESULTS_PATH)

policies = df[df["policy"] != "full_context"].copy()

SETTING_COLS = ["needle_id", "ram_attention_penalty", "ssd_attention_penalty"]

BASE_COLS = [
    "needle_id",
    "policy",
    "ram_attention_penalty",
    "ssd_attention_penalty",
    "perplexity",
    "avg_logprob",
    "vram_attention_ratio",
    "swap_total_gb",
    "predicted_text_greedy_per_step",
]


print("\n=== Winners per needle and penalty setting ===")

winners = (
    policies.sort_values(SETTING_COLS + ["perplexity"], ascending=[True, True, True, True])
    .groupby(SETTING_COLS, as_index=False)
    .first()
)

print(winners[BASE_COLS])


print("\n=== Overall win count ===")
print(winners["policy"].value_counts())


print("\n=== Win count by needle ===")
print(winners.groupby(["needle_id", "policy"]).size().unstack(fill_value=0))


print("\n=== Aggregate by policy ===")

print(
    policies.groupby("policy")
    .agg(
        mean_ppl=("perplexity", "mean"),
        median_ppl=("perplexity", "median"),
        min_ppl=("perplexity", "min"),
        max_ppl=("perplexity", "max"),
        mean_avg_logprob=("avg_logprob", "mean"),
        mean_ratio=("vram_attention_ratio", "mean"),
        mean_swap_gb=("swap_total_gb", "mean"),
    )
    .sort_values("mean_ppl")
)


print("\n=== Strong penalty settings only: SSD <= -6 ===")

strong = policies[policies["ssd_attention_penalty"] <= -6.0].copy()

strong_winners = (
    strong.sort_values(SETTING_COLS + ["perplexity"], ascending=[True, True, True, True])
    .groupby(SETTING_COLS, as_index=False)
    .first()
)

print("\nStrong win count:")
print(strong_winners["policy"].value_counts())

print("\nStrong win count by needle:")
print(strong_winners.groupby(["needle_id", "policy"]).size().unstack(fill_value=0))

print("\nStrong aggregate:")
print(
    strong.groupby("policy")
    .agg(
        mean_ppl=("perplexity", "mean"),
        median_ppl=("perplexity", "median"),
        mean_avg_logprob=("avg_logprob", "mean"),
        mean_ratio=("vram_attention_ratio", "mean"),
        mean_swap_gb=("swap_total_gb", "mean"),
    )
    .sort_values("mean_ppl")
)


print("\n=== Best row per policy ===")

print(
    policies.sort_values(["policy", "perplexity"], ascending=[True, True])
    .groupby("policy", as_index=False)
    .first()[BASE_COLS]
)


print("\n=== Worst row per policy ===")

print(
    policies.sort_values(["policy", "perplexity"], ascending=[True, False])
    .groupby("policy", as_index=False)
    .first()[BASE_COLS]
)


print("\n=== Full context quality by needle ===")

print(
    df[df["policy"] == "full_context"]
    .groupby("needle_id")
    .agg(
        mean_ppl=("perplexity", "mean"),
        median_ppl=("perplexity", "median"),
        greedy_examples=("predicted_text_greedy_per_step", lambda x: sorted(set(x))[:5]),
    )
)
