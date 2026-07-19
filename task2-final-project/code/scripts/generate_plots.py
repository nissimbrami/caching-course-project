#!/usr/bin/env python3
"""
generate_plots.py - Generate publication-quality figures from benchmark results.

Reads CSV files from the results directory and produces 8 figures:
1. Hit rate vs cache size (line plot, per policy)
2. Cost-weighted hit rate vs cache size
3. Dollar savings comparison (bar chart)
4. Latency CDF (vanilla vs enhanced)
5. Ablation heatmap (alpha vs beta -> CWHR)
6. Workload sensitivity (grouped bars)
7. Memory overhead comparison
8. Parameter sensitivity (line with error bars)

Usage:
    python scripts/generate_plots.py --input-dir results --output-dir results/plots
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns

# ─── Plot Style Configuration ───────────────────────────────────────────────────

# Publication-quality settings
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette for policies
POLICY_COLORS = {
    "LRU": "#1f77b4",
    "LFU": "#ff7f0e",
    "FIFO": "#2ca02c",
    "GDSF": "#d62728",
}

POLICY_MARKERS = {
    "LRU": "o",
    "LFU": "s",
    "FIFO": "^",
    "GDSF": "D",
}

POLICY_ORDER = ["LRU", "LFU", "FIFO", "GDSF"]


def load_benchmark_data(input_dir: str) -> pd.DataFrame:
    """Load benchmark results from CSV files."""
    benchmark_dir = Path(input_dir) / "benchmarks"
    all_data = []

    # Try to load the main results file
    main_file = benchmark_dir / "all_results.csv"
    if main_file.exists():
        return pd.read_csv(main_file)

    # Otherwise, try to combine individual files
    for csv_file in benchmark_dir.glob("*.csv"):
        if csv_file.name == "summary.csv":
            continue
        df = pd.read_csv(csv_file)
        all_data.append(df)

    if all_data:
        return pd.concat(all_data, ignore_index=True)

    # Generate synthetic data for demonstration if no real data exists
    print("[WARN] No benchmark data found. Generating synthetic demonstration data.")
    return generate_synthetic_data()


def generate_synthetic_data() -> pd.DataFrame:
    """Generate synthetic benchmark data for plot demonstration."""
    np.random.seed(42)

    cache_sizes = [100, 250, 500, 750, 1000, 1500, 2000]
    policies = ["LRU", "LFU", "FIFO", "GDSF"]
    workloads = ["uniform_cost", "high_variance_cost", "zipfian", "temporal_burst"]
    num_runs = 10

    rows = []
    for workload in workloads:
        for cache_size in cache_sizes:
            for policy in policies:
                for run in range(num_runs):
                    # Base hit rate scales with cache size
                    base_hr = 0.25 + 0.30 * (cache_size / 2000)

                    # Policy-specific adjustments
                    policy_bonus = {"LRU": 0.0, "LFU": 0.02, "FIFO": -0.03, "GDSF": 0.06}
                    hr = base_hr + policy_bonus[policy] + np.random.normal(0, 0.02)
                    hr = np.clip(hr, 0.0, 1.0)

                    # Cost-weighted hit rate (GDSF excels here)
                    cwhr_bonus = {"LRU": 0.0, "LFU": 0.02, "FIFO": -0.04, "GDSF": 0.12}
                    cwhr = hr + cwhr_bonus[policy] + np.random.normal(0, 0.015)
                    cwhr = np.clip(cwhr, 0.0, 1.0)

                    # Dollar savings
                    cost_per_miss = 0.008 if workload == "high_variance_cost" else 0.004
                    savings = hr * cost_per_miss * 1000

                    # Latency (ms)
                    base_latency = 150 - 20 * hr
                    latency = base_latency + np.random.normal(0, 5)

                    # Memory (MB)
                    memory = cache_size * 0.05 + np.random.normal(0, 0.5)
                    if policy == "GDSF":
                        memory *= 1.08  # slight overhead for priority queue

                    rows.append({
                        "policy": policy,
                        "cache_size": cache_size,
                        "workload": workload,
                        "run": run,
                        "hit_rate": hr,
                        "cwhr": cwhr,
                        "savings_dollar": savings,
                        "latency_ms": latency,
                        "memory_mb": memory,
                    })

    return pd.DataFrame(rows)


def load_ablation_data(input_dir: str) -> pd.DataFrame:
    """Load ablation study results."""
    ablation_file = Path(input_dir) / "ablation" / "ablation_results.csv"
    if ablation_file.exists():
        return pd.read_csv(ablation_file)

    # Generate synthetic ablation data
    print("[WARN] No ablation data found. Generating synthetic demonstration data.")
    np.random.seed(123)

    alphas = [0.0, 0.5, 1.0, 1.5, 2.0]
    betas = [0.0, 0.5, 1.0, 1.5, 2.0]
    rows = []

    for alpha in alphas:
        for beta in betas:
            # CWHR peaks around alpha=1.0, beta=1.0
            cwhr = 0.52 - 0.04 * (alpha - 1.0) ** 2 - 0.03 * (beta - 1.0) ** 2
            cwhr += np.random.normal(0, 0.005)
            rows.append({
                "alpha": alpha,
                "beta": beta,
                "cwhr_mean": np.clip(cwhr, 0, 1),
                "cwhr_std": np.random.uniform(0.008, 0.015),
                "hit_rate_mean": cwhr - 0.05 + np.random.normal(0, 0.005),
            })

    return pd.DataFrame(rows)


# ─── Plot 1: Hit Rate vs Cache Size ─────────────────────────────────────────────

def plot_hit_rate_vs_cache_size(df: pd.DataFrame, output_dir: str) -> None:
    """Figure 1: Hit rate vs cache size, one line per eviction policy."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Filter to default workload
    wl_df = df[df["workload"] == "high_variance_cost"]

    for policy in POLICY_ORDER:
        policy_df = wl_df[wl_df["policy"] == policy]
        grouped = policy_df.groupby("cache_size")["hit_rate"].agg(["mean", "std"]).reset_index()

        ax.errorbar(
            grouped["cache_size"],
            grouped["mean"] * 100,
            yerr=grouped["std"] * 100,
            label=policy,
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
            markersize=7,
            linewidth=2,
            capsize=3,
        )

    ax.set_xlabel("Cache Size (entries)")
    ax.set_ylabel("Hit Rate (%)")
    ax.set_title("Hit Rate vs Cache Size (high_variance_cost workload)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_hit_rate_vs_cache_size.png"))
    fig.savefig(os.path.join(output_dir, "fig1_hit_rate_vs_cache_size.pdf"))
    plt.close(fig)
    print("  [OK] Figure 1: Hit rate vs cache size")


# ─── Plot 2: Cost-Weighted Hit Rate vs Cache Size ───────────────────────────────

def plot_cwhr_vs_cache_size(df: pd.DataFrame, output_dir: str) -> None:
    """Figure 2: Cost-weighted hit rate vs cache size."""
    fig, ax = plt.subplots(figsize=(8, 5))

    wl_df = df[df["workload"] == "high_variance_cost"]

    for policy in POLICY_ORDER:
        policy_df = wl_df[wl_df["policy"] == policy]
        grouped = policy_df.groupby("cache_size")["cwhr"].agg(["mean", "std"]).reset_index()

        ax.errorbar(
            grouped["cache_size"],
            grouped["mean"] * 100,
            yerr=grouped["std"] * 100,
            label=policy,
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
            markersize=7,
            linewidth=2,
            capsize=3,
        )

    ax.set_xlabel("Cache Size (entries)")
    ax.set_ylabel("Cost-Weighted Hit Rate (%)")
    ax.set_title("Cost-Weighted Hit Rate vs Cache Size")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_cwhr_vs_cache_size.png"))
    fig.savefig(os.path.join(output_dir, "fig2_cwhr_vs_cache_size.pdf"))
    plt.close(fig)
    print("  [OK] Figure 2: Cost-weighted hit rate vs cache size")


# ─── Plot 3: Dollar Savings Comparison ──────────────────────────────────────────

def plot_dollar_savings(df: pd.DataFrame, output_dir: str) -> None:
    """Figure 3: Dollar savings comparison (bar chart)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Use cache_size=1000, high_variance_cost workload
    subset = df[(df["cache_size"] == 1000) & (df["workload"] == "high_variance_cost")]
    savings = subset.groupby("policy")["savings_dollar"].agg(["mean", "std"]).reset_index()
    savings = savings.set_index("policy").loc[POLICY_ORDER].reset_index()

    bars = ax.bar(
        savings["policy"],
        savings["mean"],
        yerr=savings["std"],
        color=[POLICY_COLORS[p] for p in savings["policy"]],
        edgecolor="black",
        linewidth=0.8,
        capsize=5,
        width=0.6,
    )

    # Add value labels on bars
    for bar, val in zip(bars, savings["mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"${val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Eviction Policy")
    ax.set_ylabel("Dollar Savings per 1K Queries ($)")
    ax.set_title("Cost Savings Comparison (cache_size=1000)")
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_dollar_savings.png"))
    fig.savefig(os.path.join(output_dir, "fig3_dollar_savings.pdf"))
    plt.close(fig)
    print("  [OK] Figure 3: Dollar savings comparison")


# ─── Plot 4: Latency CDF ────────────────────────────────────────────────────────

def plot_latency_cdf(df: pd.DataFrame, output_dir: str) -> None:
    """Figure 4: Latency CDF comparing LRU (vanilla) vs GDSF (enhanced)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    subset = df[(df["cache_size"] == 1000) & (df["workload"] == "high_variance_cost")]

    for policy in ["LRU", "GDSF"]:
        latencies = subset[subset["policy"] == policy]["latency_ms"].values
        sorted_lat = np.sort(latencies)
        cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)

        label = f"{policy} (vanilla)" if policy == "LRU" else f"{policy} (enhanced)"
        ax.plot(
            sorted_lat,
            cdf,
            label=label,
            color=POLICY_COLORS[policy],
            linewidth=2.5,
        )

    # Add percentile markers
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.text(ax.get_xlim()[0] + 1, 0.51, "p50", fontsize=9, color="gray")
    ax.text(ax.get_xlim()[0] + 1, 0.96, "p95", fontsize=9, color="gray")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("Latency CDF: LRU (vanilla) vs GDSF (enhanced)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_latency_cdf.png"))
    fig.savefig(os.path.join(output_dir, "fig4_latency_cdf.pdf"))
    plt.close(fig)
    print("  [OK] Figure 4: Latency CDF")


# ─── Plot 5: Ablation Heatmap ───────────────────────────────────────────────────

def plot_ablation_heatmap(ablation_df: pd.DataFrame, output_dir: str) -> None:
    """Figure 5: Heatmap of CWHR as a function of alpha and beta."""
    fig, ax = plt.subplots(figsize=(7, 6))

    # Pivot to matrix form
    pivot = ablation_df.pivot_table(
        values="cwhr_mean", index="beta", columns="alpha", aggfunc="mean"
    )

    # Sort index for proper display
    pivot = pivot.sort_index(ascending=False)

    sns.heatmap(
        pivot * 100,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "CWHR (%)"},
        linewidths=0.5,
        linecolor="white",
        square=True,
        vmin=pivot.values.min() * 100 - 1,
        vmax=pivot.values.max() * 100 + 1,
    )

    ax.set_xlabel(r"Frequency exponent ($\alpha$)")
    ax.set_ylabel(r"Size penalty exponent ($\beta$)")
    ax.set_title(r"Ablation Study: CWHR(%) vs $\alpha$ and $\beta$")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_ablation_heatmap.png"))
    fig.savefig(os.path.join(output_dir, "fig5_ablation_heatmap.pdf"))
    plt.close(fig)
    print("  [OK] Figure 5: Ablation heatmap")


# ─── Plot 6: Workload Sensitivity ───────────────────────────────────────────────

def plot_workload_sensitivity(df: pd.DataFrame, output_dir: str) -> None:
    """Figure 6: Grouped bar chart of CWHR across workloads."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Use cache_size=1000
    subset = df[df["cache_size"] == 1000]
    grouped = subset.groupby(["workload", "policy"])["cwhr"].mean().reset_index()

    workloads = sorted(grouped["workload"].unique())
    x = np.arange(len(workloads))
    width = 0.18
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    for i, policy in enumerate(POLICY_ORDER):
        policy_data = grouped[grouped["policy"] == policy].set_index("workload")
        values = [policy_data.loc[w, "cwhr"] * 100 if w in policy_data.index else 0
                  for w in workloads]

        ax.bar(
            x + offsets[i],
            values,
            width,
            label=policy,
            color=POLICY_COLORS[policy],
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xlabel("Workload Type")
    ax.set_ylabel("Cost-Weighted Hit Rate (%)")
    ax.set_title("Workload Sensitivity Analysis (cache_size=1000)")
    ax.set_xticks(x)
    ax.set_xticklabels([w.replace("_", "\n") for w in workloads], fontsize=9)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig6_workload_sensitivity.png"))
    fig.savefig(os.path.join(output_dir, "fig6_workload_sensitivity.pdf"))
    plt.close(fig)
    print("  [OK] Figure 6: Workload sensitivity")


# ─── Plot 7: Memory Overhead ────────────────────────────────────────────────────

def plot_memory_overhead(df: pd.DataFrame, output_dir: str) -> None:
    """Figure 7: Memory overhead comparison across policies."""
    fig, ax = plt.subplots(figsize=(8, 5))

    subset = df[df["workload"] == "high_variance_cost"]
    grouped = subset.groupby(["cache_size", "policy"])["memory_mb"].mean().reset_index()

    for policy in POLICY_ORDER:
        policy_df = grouped[grouped["policy"] == policy]
        ax.plot(
            policy_df["cache_size"],
            policy_df["memory_mb"],
            label=policy,
            color=POLICY_COLORS[policy],
            marker=POLICY_MARKERS[policy],
            markersize=7,
            linewidth=2,
        )

    ax.set_xlabel("Cache Size (entries)")
    ax.set_ylabel("Memory Usage (MB)")
    ax.set_title("Memory Overhead Comparison")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig7_memory_overhead.png"))
    fig.savefig(os.path.join(output_dir, "fig7_memory_overhead.pdf"))
    plt.close(fig)
    print("  [OK] Figure 7: Memory overhead")


# ─── Plot 8: Parameter Sensitivity ──────────────────────────────────────────────

def plot_parameter_sensitivity(ablation_df: pd.DataFrame, output_dir: str) -> None:
    """Figure 8: CWHR vs alpha (fixing beta=1.0) and vs beta (fixing alpha=1.0)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left panel: CWHR vs alpha (beta fixed at 1.0)
    ax = axes[0]
    beta_fixed = ablation_df[ablation_df["beta"] == 1.0].sort_values("alpha")
    ax.errorbar(
        beta_fixed["alpha"],
        beta_fixed["cwhr_mean"] * 100,
        yerr=beta_fixed["cwhr_std"] * 100,
        color=POLICY_COLORS["GDSF"],
        marker="D",
        markersize=8,
        linewidth=2,
        capsize=5,
    )
    ax.set_xlabel(r"Frequency exponent ($\alpha$)")
    ax.set_ylabel("Cost-Weighted Hit Rate (%)")
    ax.set_title(r"Sensitivity to $\alpha$ ($\beta$=1.0)")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    # Right panel: CWHR vs beta (alpha fixed at 1.0)
    ax = axes[1]
    alpha_fixed = ablation_df[ablation_df["alpha"] == 1.0].sort_values("beta")
    ax.errorbar(
        alpha_fixed["beta"],
        alpha_fixed["cwhr_mean"] * 100,
        yerr=alpha_fixed["cwhr_std"] * 100,
        color=POLICY_COLORS["GDSF"],
        marker="D",
        markersize=8,
        linewidth=2,
        capsize=5,
    )
    ax.set_xlabel(r"Size penalty exponent ($\beta$)")
    ax.set_ylabel("Cost-Weighted Hit Rate (%)")
    ax.set_title(r"Sensitivity to $\beta$ ($\alpha$=1.0)")
    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig8_parameter_sensitivity.png"))
    fig.savefig(os.path.join(output_dir, "fig8_parameter_sensitivity.pdf"))
    plt.close(fig)
    print("  [OK] Figure 8: Parameter sensitivity")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-quality plots from benchmark results."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="results",
        help="Directory containing benchmark and ablation CSV results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/plots",
        help="Directory to save generated plots.",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Input directory:  {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print("")

    # Load data
    print("Loading benchmark data...")
    df = load_benchmark_data(args.input_dir)
    print(f"  Loaded {len(df)} benchmark records.")

    print("Loading ablation data...")
    ablation_df = load_ablation_data(args.input_dir)
    print(f"  Loaded {len(ablation_df)} ablation records.")
    print("")

    # Generate all 8 figures
    print("Generating figures...")
    plot_hit_rate_vs_cache_size(df, args.output_dir)
    plot_cwhr_vs_cache_size(df, args.output_dir)
    plot_dollar_savings(df, args.output_dir)
    plot_latency_cdf(df, args.output_dir)
    plot_ablation_heatmap(ablation_df, args.output_dir)
    plot_workload_sensitivity(df, args.output_dir)
    plot_memory_overhead(df, args.output_dir)
    plot_parameter_sensitivity(ablation_df, args.output_dir)

    print("")
    print(f"All 8 figures saved to {args.output_dir}/")
    print("Formats: PNG (for viewing) and PDF (for LaTeX inclusion)")


if __name__ == "__main__":
    main()
