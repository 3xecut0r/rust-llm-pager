from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_PATH = Path("bench/multi_needle_soft_masked_grid_results.csv")
PLOTS_DIR = Path("bench/plots")


def main():
    """Render win-rate and perplexity charts from the multi-needle grid results CSV."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_PATH)
    policies = df[df["policy"] != "full_context"].copy()

    setting_cols = ["needle_id", "ram_attention_penalty", "ssd_attention_penalty"]

    winners = (
        policies.sort_values(setting_cols + ["perplexity"], ascending=[True, True, True, True])
        .groupby(setting_cols, as_index=False)
        .first()
    )

    win_counts = winners["policy"].value_counts()

    plt.figure(figsize=(9, 5))
    plt.bar(win_counts.index, win_counts.values)
    plt.xlabel("Policy")
    plt.ylabel("Wins")
    plt.title("Multi-needle soft masked grid: wins by policy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)

    out_path = PLOTS_DIR / "multi_needle_soft_masked_grid_wins.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")

    aggregate = (
        policies.groupby("policy")
        .agg(
            mean_ppl=("perplexity", "mean"),
            median_ppl=("perplexity", "median"),
            mean_ratio=("vram_attention_ratio", "mean"),
            mean_swap_gb=("swap_total_gb", "mean"),
        )
        .reset_index()
        .sort_values("mean_ppl")
    )

    plt.figure(figsize=(9, 5))
    plt.bar(aggregate["policy"], aggregate["mean_ppl"])
    plt.xlabel("Policy")
    plt.ylabel("Mean perplexity")
    plt.title("Multi-needle soft masked grid: mean perplexity")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)

    out_path = PLOTS_DIR / "multi_needle_soft_masked_grid_mean_ppl.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")

    plt.figure(figsize=(9, 5))
    plt.bar(aggregate["policy"], aggregate["median_ppl"])
    plt.xlabel("Policy")
    plt.ylabel("Median perplexity")
    plt.title("Multi-needle soft masked grid: median perplexity")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)

    out_path = PLOTS_DIR / "multi_needle_soft_masked_grid_median_ppl.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")

    strong = policies[policies["ssd_attention_penalty"] <= -6.0].copy()

    strong_winners = (
        strong.sort_values(setting_cols + ["perplexity"], ascending=[True, True, True, True])
        .groupby(setting_cols, as_index=False)
        .first()
    )

    strong_win_counts = strong_winners["policy"].value_counts()

    plt.figure(figsize=(9, 5))
    plt.bar(strong_win_counts.index, strong_win_counts.values)
    plt.xlabel("Policy")
    plt.ylabel("Wins")
    plt.title("Multi-needle strong SSD penalties: wins by policy")
    plt.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=20)

    out_path = PLOTS_DIR / "multi_needle_soft_masked_grid_strong_wins.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")

    by_needle = winners.groupby(["needle_id", "policy"]).size().unstack(fill_value=0)

    ax = by_needle.plot(kind="bar", figsize=(11, 6))
    ax.set_xlabel("Needle")
    ax.set_ylabel("Wins")
    ax.set_title("Multi-needle wins by prompt")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")

    out_path = PLOTS_DIR / "multi_needle_soft_masked_grid_wins_by_needle.png"
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
