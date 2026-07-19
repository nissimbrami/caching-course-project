#!/usr/bin/env python3
"""
run_ablation.py - Run the GDSF parameter ablation study.

Sweeps over combinations of alpha (frequency exponent) and beta (cost exponent)
to measure their effect on cost-weighted hit rate (CWHR).

Configuration:
    - alpha values: [0.0, 0.5, 1.0, 1.5, 2.0]
    - beta values:  [0.0, 0.5, 1.0, 1.5, 2.0]
    - Fixed workload: high_variance_cost
    - Fixed cache size: 1000
    - Repetitions: 10 runs per (alpha, beta) pair

Output:
    - ablation_results.csv with columns:
      alpha, beta, run, hit_rate, cwhr, savings_dollar, latency_ms

Usage:
    python scripts/run_ablation.py --cache-size 1000 --num-runs 10 --output-dir results/ablation
"""

import argparse
import os
import sys
import time
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def estimate_query_cost(num_tokens: int, model: str = "gpt-3.5-turbo") -> float:
    """Estimate the dollar cost of a query based on token count.

    Pricing (approximate, per 1K tokens):
        gpt-3.5-turbo: $0.0015 input, $0.002 output
        gpt-4: $0.03 input, $0.06 output
    """
    pricing = {
        "gpt-3.5-turbo": {"input": 0.0015 / 1000, "output": 0.002 / 1000},
        "gpt-4": {"input": 0.03 / 1000, "output": 0.06 / 1000},
    }
    rates = pricing.get(model, pricing["gpt-3.5-turbo"])
    # Assume roughly equal input/output tokens
    return num_tokens * (rates["input"] + rates["output"]) / 2


def generate_workload(
    workload_type: str, num_queries: int, seed: int = 42
) -> list[dict]:
    """Generate a synthetic workload with varying costs and access patterns.

    Each query has:
        - query_id: unique identifier
        - tokens: token count (determines cost)
        - cost: dollar cost estimate
        - size: memory footprint in bytes
        - popularity: access probability weight
    """
    rng = np.random.default_rng(seed)

    queries = []
    unique_items = num_queries // 5  # 20% unique items, rest are repeats

    if workload_type == "high_variance_cost":
        # High variance in token counts (and thus costs)
        token_counts = rng.lognormal(mean=6.0, sigma=1.5, size=unique_items).astype(int)
        token_counts = np.clip(token_counts, 50, 10000)
        # Zipf popularity distribution
        popularity = 1.0 / np.arange(1, unique_items + 1) ** 0.8
        popularity /= popularity.sum()
    elif workload_type == "uniform_cost":
        # Uniform token counts
        token_counts = rng.integers(200, 600, size=unique_items)
        popularity = np.ones(unique_items) / unique_items
    elif workload_type == "zipfian":
        # Moderate cost variance, strong Zipf popularity
        token_counts = rng.integers(100, 2000, size=unique_items)
        popularity = 1.0 / np.arange(1, unique_items + 1) ** 1.2
        popularity /= popularity.sum()
    elif workload_type == "temporal_burst":
        # Bursty access with shifting hot set
        token_counts = rng.lognormal(mean=5.5, sigma=1.0, size=unique_items).astype(int)
        token_counts = np.clip(token_counts, 50, 8000)
        popularity = np.ones(unique_items) / unique_items  # will be overridden per burst
    else:
        raise ValueError(f"Unknown workload type: {workload_type}")

    # Build the catalog of unique items
    catalog = []
    for i in range(unique_items):
        tokens = int(token_counts[i])
        catalog.append({
            "query_id": i,
            "tokens": tokens,
            "cost": estimate_query_cost(tokens),
            "size": tokens * 4 + 512,  # approximate memory in bytes
        })

    # Generate query stream based on popularity
    if workload_type == "temporal_burst":
        # Divide into bursts with different hot sets
        burst_size = num_queries // 4
        query_stream = []
        for burst in range(4):
            # Each burst has a different hot set (25% of items)
            hot_start = (burst * unique_items // 4) % unique_items
            hot_indices = np.arange(hot_start, hot_start + unique_items // 4) % unique_items
            burst_popularity = np.zeros(unique_items)
            burst_popularity[hot_indices] = 1.0
            burst_popularity /= burst_popularity.sum()
            indices = rng.choice(unique_items, size=burst_size, p=burst_popularity)
            query_stream.extend(indices)
    else:
        query_stream = rng.choice(unique_items, size=num_queries, p=popularity)

    # Build final workload
    for idx in query_stream:
        queries.append(catalog[idx].copy())

    return queries


class GDSFCache:
    """Greedy Dual-Size Frequency (GDSF) cache implementation.

    Priority formula:
        priority(entry) = L + freq(entry)^alpha * cost(entry)^beta / size(entry)

    Where:
        L = clock value (aging factor)
        freq = access frequency since insertion
        cost = dollar cost of regenerating the response
        size = memory footprint
        alpha = frequency sensitivity exponent
        beta = cost sensitivity exponent
    """

    def __init__(self, max_size: int, alpha: float = 1.0, beta: float = 1.0):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.clock = 0.0  # Aging clock L

        # Cache storage: query_id -> entry dict
        self.cache: dict[int, dict] = {}
        # Priority values: query_id -> priority
        self.priorities: dict[int, float] = {}
        # Access frequencies: query_id -> count
        self.frequencies: dict[int, int] = {}

    def _compute_priority(self, entry: dict) -> float:
        """Compute GDSF priority for an entry.

        Formula: Priority(i) = Clock + (freq^alpha * cost^beta) / size
        This matches the main eviction_manager.py implementation exactly.
        """
        freq = self.frequencies.get(entry["query_id"], 1)
        cost = entry["cost"]
        size = entry["size"]

        # Match main implementation: freq^alpha * cost^beta / size
        freq_factor = freq ** self.alpha
        cost_factor = cost ** self.beta
        size_factor = max(size, 1)

        priority = self.clock + (freq_factor * cost_factor) / size_factor
        return priority

    def access(self, query: dict) -> bool:
        """Process a cache access. Returns True if hit, False if miss."""
        query_id = query["query_id"]

        if query_id in self.cache:
            # Cache hit: update frequency and priority
            self.frequencies[query_id] += 1
            self.priorities[query_id] = self._compute_priority(self.cache[query_id])
            return True

        # Cache miss: insert (possibly after eviction)
        self._insert(query)
        return False

    def _insert(self, query: dict) -> None:
        """Insert a new entry, evicting if necessary."""
        query_id = query["query_id"]

        # Evict if at capacity
        while len(self.cache) >= self.max_size:
            self._evict()

        # Insert new entry
        self.cache[query_id] = query
        self.frequencies[query_id] = 1
        self.priorities[query_id] = self._compute_priority(query)

    def _evict(self) -> None:
        """Evict the entry with the lowest priority."""
        if not self.priorities:
            return

        # Find minimum priority entry
        victim_id = min(self.priorities, key=self.priorities.get)

        # Update clock to victim's priority (aging mechanism)
        self.clock = self.priorities[victim_id]

        # Remove victim
        del self.cache[victim_id]
        del self.priorities[victim_id]
        del self.frequencies[victim_id]


class BaselineCache:
    """Simple LRU/LFU/FIFO baseline for comparison (not used in ablation)."""

    def __init__(self, max_size: int, policy: str = "LRU"):
        self.max_size = max_size
        self.policy = policy
        self.cache: dict[int, dict] = {}
        self.access_order: list[int] = []  # for LRU/FIFO
        self.frequencies: dict[int, int] = {}  # for LFU

    def access(self, query: dict) -> bool:
        query_id = query["query_id"]

        if query_id in self.cache:
            self.frequencies[query_id] = self.frequencies.get(query_id, 0) + 1
            if self.policy == "LRU":
                self.access_order.remove(query_id)
                self.access_order.append(query_id)
            return True

        # Miss - evict and insert
        while len(self.cache) >= self.max_size:
            self._evict()

        self.cache[query_id] = query
        self.access_order.append(query_id)
        self.frequencies[query_id] = 1
        return False

    def _evict(self) -> None:
        if self.policy in ("LRU", "FIFO"):
            victim_id = self.access_order.pop(0)
        elif self.policy == "LFU":
            victim_id = min(self.frequencies, key=self.frequencies.get)
        else:
            victim_id = self.access_order.pop(0)

        del self.cache[victim_id]
        if victim_id in self.frequencies:
            del self.frequencies[victim_id]
        if victim_id in self.access_order:
            self.access_order.remove(victim_id)


def run_single_experiment(
    workload: list[dict],
    cache_size: int,
    alpha: float,
    beta: float,
    seed: int,
) -> dict:
    """Run a single GDSF experiment and return metrics.

    Returns:
        dict with keys: hit_rate, cwhr, savings_dollar, latency_ms
    """
    cache = GDSFCache(max_size=cache_size, alpha=alpha, beta=beta)

    hits = 0
    total = 0
    cost_weighted_hits = 0.0
    total_cost = 0.0
    total_savings = 0.0

    # Simulate latency: hit = fast lookup, miss = full LLM call
    hit_latency_base = 5.0  # ms
    miss_latency_base = 200.0  # ms
    latencies = []

    rng = np.random.default_rng(seed)

    for query in workload:
        total += 1
        total_cost += query["cost"]

        is_hit = cache.access(query)

        if is_hit:
            hits += 1
            cost_weighted_hits += query["cost"]
            total_savings += query["cost"]
            latency = hit_latency_base + rng.exponential(2.0)
        else:
            latency = miss_latency_base + rng.exponential(50.0)

        latencies.append(latency)

    hit_rate = hits / total if total > 0 else 0.0
    cwhr = cost_weighted_hits / total_cost if total_cost > 0 else 0.0
    # Savings per 1K queries
    savings_dollar = (total_savings / total) * 1000 if total > 0 else 0.0
    avg_latency = np.mean(latencies)

    return {
        "hit_rate": hit_rate,
        "cwhr": cwhr,
        "savings_dollar": savings_dollar,
        "latency_ms": avg_latency,
    }


def run_ablation_study(
    cache_size: int = 1000,
    num_runs: int = 10,
    num_queries: int = 5000,
    workload_type: str = "high_variance_cost",
) -> pd.DataFrame:
    """Run full ablation study sweeping alpha and beta.

    Args:
        cache_size: Fixed cache size for all experiments.
        num_runs: Number of repetitions per (alpha, beta) pair.
        num_queries: Number of queries per workload instance.
        workload_type: Type of workload to generate.

    Returns:
        DataFrame with columns: alpha, beta, run, hit_rate, cwhr, savings_dollar, latency_ms
    """
    alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0]
    beta_values = [0.0, 0.5, 1.0, 1.5, 2.0]

    total_experiments = len(alpha_values) * len(beta_values) * num_runs
    print(f"Running ablation study:")
    print(f"  Alpha values: {alpha_values}")
    print(f"  Beta values:  {beta_values}")
    print(f"  Cache size:   {cache_size}")
    print(f"  Num runs:     {num_runs}")
    print(f"  Workload:     {workload_type}")
    print(f"  Queries/run:  {num_queries}")
    print(f"  Total experiments: {total_experiments}")
    print("")

    results = []

    # Pre-generate workloads with fixed seeds per run (independent of alpha/beta)
    # This ensures all (alpha, beta) configurations see the SAME workloads,
    # isolating the effect of parameters from workload randomness.
    print("Pre-generating workloads...")
    workloads = []
    for run in range(num_runs):
        workload_seed = 1000 + run  # Fixed seed per run, independent of params
        workloads.append(generate_workload(
            workload_type=workload_type,
            num_queries=num_queries,
            seed=workload_seed,
        ))

    with tqdm(total=total_experiments, desc="Ablation sweep") as pbar:
        for alpha, beta in product(alpha_values, beta_values):
            for run in range(num_runs):
                # Use pre-generated workload (same across all alpha/beta)
                workload = workloads[run]

                # Latency RNG seed: fixed per run, independent of params
                latency_seed = 5000 + run

                # Run experiment
                metrics = run_single_experiment(
                    workload=workload,
                    cache_size=cache_size,
                    alpha=alpha,
                    beta=beta,
                    seed=latency_seed,
                )

                results.append({
                    "alpha": alpha,
                    "beta": beta,
                    "run": run,
                    **metrics,
                })

                pbar.update(1)

    return pd.DataFrame(results)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics from ablation results."""
    summary = df.groupby(["alpha", "beta"]).agg(
        hit_rate_mean=("hit_rate", "mean"),
        hit_rate_std=("hit_rate", "std"),
        cwhr_mean=("cwhr", "mean"),
        cwhr_std=("cwhr", "std"),
        savings_mean=("savings_dollar", "mean"),
        savings_std=("savings_dollar", "std"),
        latency_mean=("latency_ms", "mean"),
        latency_std=("latency_ms", "std"),
    ).reset_index()

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run GDSF parameter ablation study (alpha x beta sweep)."
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=int(os.environ.get("CACHE_SIZE", "1000")),
        help="Cache size (number of entries). Default: 1000",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=int(os.environ.get("NUM_RUNS", "10")),
        help="Number of runs per (alpha, beta) pair. Default: 10",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=5000,
        help="Number of queries per workload instance. Default: 5000",
    )
    parser.add_argument(
        "--workload",
        type=str,
        default="high_variance_cost",
        choices=["uniform_cost", "high_variance_cost", "zipfian", "temporal_burst"],
        help="Workload type. Default: high_variance_cost",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "results/ablation"),
        help="Output directory for results. Default: results/ablation",
    )
    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()

    # Run ablation study
    results_df = run_ablation_study(
        cache_size=args.cache_size,
        num_runs=args.num_runs,
        num_queries=args.num_queries,
        workload_type=args.workload,
    )

    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f}s")

    # Save raw results
    raw_path = os.path.join(args.output_dir, "ablation_results_raw.csv")
    results_df.to_csv(raw_path, index=False)
    print(f"Raw results saved to: {raw_path}")

    # Compute and save summary
    summary_df = compute_summary(results_df)
    summary_path = os.path.join(args.output_dir, "ablation_results.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to: {summary_path}")

    # Print best configuration
    best_idx = summary_df["cwhr_mean"].idxmax()
    best = summary_df.iloc[best_idx]
    print(f"\nBest configuration:")
    print(f"  alpha = {best['alpha']:.1f}, beta = {best['beta']:.1f}")
    print(f"  CWHR  = {best['cwhr_mean']*100:.2f}% (+/- {best['cwhr_std']*100:.2f}%)")
    print(f"  Hit Rate = {best['hit_rate_mean']*100:.2f}%")
    print(f"  Savings  = ${best['savings_mean']:.2f}/1K queries")


if __name__ == "__main__":
    main()
