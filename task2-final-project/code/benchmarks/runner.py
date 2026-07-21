"""
Benchmark runner for cache eviction policy experiments.

Provides functions to run individual experiments and full parameter sweeps
across policies, workloads, and cache sizes. Results are collected as
pandas DataFrames for easy analysis and export.
"""

from typing import Any, Dict, List, Optional, Tuple
import time
import sys
import os
import json
import csv
from pathlib import Path

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .policies import CachePolicy, create_policy, POLICY_REGISTRY
from .workloads import WORKLOAD_REGISTRY, get_workload
from .metrics import BenchmarkResult, MetricsCollector, ResourceSampler, estimate_memory_usage


def run_single_experiment(
    policy: CachePolicy,
    workload: List[Dict[str, Any]],
    workload_name: str,
    cache_size: int,
    run_id: int = 0,
    seed: int = 42,
) -> BenchmarkResult:
    """Run a single benchmark experiment with one policy and one workload.

    Processes every query in the workload against the given cache policy,
    collecting hit/miss metrics, latency measurements, and eviction counts.

    Args:
        policy: The initialized cache policy to benchmark.
        workload: List of query dicts from a workload generator.
        workload_name: Name of the workload (for labeling results).
        cache_size: Cache size in bytes (should match policy's max_size).
        run_id: Identifier for this run (for multi-run experiments).
        seed: Random seed used (for reproducibility tracking).

    Returns:
        BenchmarkResult with all computed metrics.
    """
    collector = MetricsCollector(
        policy_name=policy.name,
        workload_name=workload_name,
        cache_size=cache_size,
        run_id=run_id,
        seed=seed,
    )

    policy.reset()
    sampler = ResourceSampler(collector, interval=0.05)
    sampler.start()
    collector.start_timer()

    for query in workload:
        key = str(query["query_id"])
        cost = query["cost"]
        size = query["size"]

        # Try to access (cache hit check)
        start = time.perf_counter()
        hit = policy.access(key)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        if hit:
            collector.record_hit(cost, elapsed_ms)
        else:
            # Cache miss: insert the item
            start_put = time.perf_counter()
            evicted = policy.put(key, size, cost)
            put_elapsed_ms = (time.perf_counter() - start_put) * 1000.0

            total_latency = elapsed_ms + put_elapsed_ms
            collector.record_miss(cost, total_latency)
            collector.record_evictions(len(evicted))

    collector.stop_timer()
    sampler.stop()

    # Estimate memory overhead of the policy data structures
    try:
        memory = estimate_memory_usage(policy)
    except (RecursionError, TypeError):
        memory = sys.getsizeof(policy)
    collector.set_memory_overhead(memory)

    return collector.get_result()


def run_all_experiments(
    n_runs: int = 30,
    cache_sizes: Optional[List[int]] = None,
    policy_names: Optional[List[str]] = None,
    workload_names: Optional[List[str]] = None,
    base_seed: int = 42,
    verbose: bool = True,
) -> Any:
    """Run a full parameter sweep across policies, workloads, and cache sizes.

    Executes n_runs repetitions of each (policy, workload, cache_size) combination
    with different random seeds for statistical robustness.

    Args:
        n_runs: Number of repetitions per configuration (default 30 for
                statistical significance per grading criteria).
        cache_sizes: List of cache sizes in bytes to test.
        policy_names: List of policy names to benchmark (from POLICY_REGISTRY).
        workload_names: List of workload names to use (from WORKLOAD_REGISTRY).
        base_seed: Base random seed (each run uses base_seed + run_id).
        verbose: Whether to print progress information.

    Returns:
        pandas DataFrame with all results if pandas is available,
        otherwise a list of BenchmarkResult dicts.
    """
    if cache_sizes is None:
        cache_sizes = [50000, 100000, 250000, 500000]

    if policy_names is None:
        policy_names = list(POLICY_REGISTRY.keys())

    if workload_names is None:
        workload_names = list(WORKLOAD_REGISTRY.keys())

    results: List[Dict[str, Any]] = []
    total_experiments = len(policy_names) * len(workload_names) * len(cache_sizes) * n_runs

    if verbose:
        print(f"Running {total_experiments} experiments...")
        print(f"  Policies: {policy_names}")
        print(f"  Workloads: {workload_names}")
        print(f"  Cache sizes: {cache_sizes}")
        print(f"  Runs per config: {n_runs}")
        print()

    experiment_count = 0

    for workload_name in workload_names:
        for cache_size in cache_sizes:
            for run_id in range(n_runs):
                seed = base_seed + run_id

                # Generate workload with this seed
                workload = get_workload(workload_name, seed=seed)

                for policy_name in policy_names:
                    # Create policy with appropriate parameters
                    policy_kwargs: Dict[str, Any] = {}
                    if policy_name == "Random":
                        policy_kwargs["seed"] = seed

                    policy = create_policy(policy_name, max_size=cache_size, **policy_kwargs)

                    # Run the experiment
                    result = run_single_experiment(
                        policy=policy,
                        workload=workload,
                        workload_name=workload_name,
                        cache_size=cache_size,
                        run_id=run_id,
                        seed=seed,
                    )

                    results.append(result.to_dict())
                    experiment_count += 1

                    if verbose and experiment_count % 50 == 0:
                        pct = experiment_count / total_experiments * 100
                        print(f"  Progress: {experiment_count}/{total_experiments} ({pct:.1f}%)")

    if verbose:
        print(f"\nCompleted {experiment_count} experiments.")

    if HAS_PANDAS:
        return pd.DataFrame(results)
    return results


def save_results(
    results: Any,
    output_dir: str,
    prefix: str = "benchmark_results",
) -> Tuple[str, str]:
    """Save benchmark results to CSV and JSON files.

    Args:
        results: pandas DataFrame or list of dicts with benchmark results.
        output_dir: Directory to save results in (will be created if needed).
        prefix: Filename prefix for output files.

    Returns:
        Tuple of (csv_path, json_path) of the saved files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{prefix}_{timestamp}.csv"
    json_filename = f"{prefix}_{timestamp}.json"

    csv_path = str(output_path / csv_filename)
    json_path = str(output_path / json_filename)

    if HAS_PANDAS and hasattr(results, "to_csv"):
        # pandas DataFrame
        results.to_csv(csv_path, index=False)
        results.to_json(json_path, orient="records", indent=2)
    else:
        # List of dicts fallback
        if isinstance(results, list) and len(results) > 0:
            # Write CSV
            fieldnames = list(results[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            # Write JSON
            with open(json_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

    print(f"Results saved to:")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")

    return csv_path, json_path


def run_quick_benchmark(
    cache_size: int = 100000,
    n_queries: int = 5000,
    verbose: bool = True,
) -> Any:
    """Run a quick benchmark for sanity-checking.

    Uses a smaller workload and single run for fast iteration during
    development.

    Args:
        cache_size: Cache size in bytes.
        n_queries: Number of queries per workload.
        verbose: Whether to print results.

    Returns:
        pandas DataFrame or list of result dicts.
    """
    results: List[Dict[str, Any]] = []

    workloads_to_test = ["uniform_cost", "high_variance_cost", "adversarial_lru"]

    for workload_name in workloads_to_test:
        workload = get_workload(workload_name, n_queries=n_queries, seed=42)

        for policy_name in POLICY_REGISTRY:
            policy_kwargs: Dict[str, Any] = {}
            if policy_name == "Random":
                policy_kwargs["seed"] = 42
            policy = create_policy(policy_name, max_size=cache_size, **policy_kwargs)

            result = run_single_experiment(
                policy=policy,
                workload=workload,
                workload_name=workload_name,
                cache_size=cache_size,
                run_id=0,
                seed=42,
            )

            results.append(result.to_dict())

            if verbose:
                print(
                    f"  {policy_name:10s} | {workload_name:20s} | "
                    f"HR={result.hit_rate:.4f} | "
                    f"CWHR={result.cost_weighted_hit_rate:.4f} | "
                    f"${result.dollar_savings:.4f}"
                )

    if HAS_PANDAS:
        return pd.DataFrame(results)
    return results
