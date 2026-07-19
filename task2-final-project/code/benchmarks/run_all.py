"""
CLI entry point for running the full benchmark suite.

Supports parallel execution, progress reporting, and flexible configuration
via command-line arguments.

Usage:
    python -m benchmarks.run_all --n-runs 30 --output-dir results/
    python -m benchmarks.run_all --workloads adversarial_lru high_variance_cost --policies LRU GDSF
    python -m benchmarks.run_all --quick
"""

import argparse
import json
import multiprocessing
import os
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from .policies import CachePolicy, create_policy, POLICY_REGISTRY
from .workloads import WORKLOAD_REGISTRY, get_workload
from .metrics import BenchmarkResult, MetricsCollector
from .runner import run_single_experiment, save_results, run_quick_benchmark


def _run_experiment_task(args: Tuple[str, str, int, int, int]) -> Dict[str, Any]:
    """Worker function for parallel experiment execution.

    Designed to be called via multiprocessing.Pool.map(). Each invocation
    creates a fresh policy and workload from the given parameters.

    Args:
        args: Tuple of (policy_name, workload_name, cache_size, run_id, seed).

    Returns:
        Dictionary of benchmark results.
    """
    policy_name, workload_name, cache_size, run_id, seed = args

    # Generate workload
    workload = get_workload(workload_name, seed=seed)

    # Create policy
    policy_kwargs: Dict[str, Any] = {}
    if policy_name == "Random":
        policy_kwargs["seed"] = seed

    policy = create_policy(policy_name, max_size=cache_size, **policy_kwargs)

    # Run experiment
    result = run_single_experiment(
        policy=policy,
        workload=workload,
        workload_name=workload_name,
        cache_size=cache_size,
        run_id=run_id,
        seed=seed,
    )

    return result.to_dict()


def build_experiment_tasks(
    policy_names: List[str],
    workload_names: List[str],
    cache_sizes: List[int],
    n_runs: int,
    base_seed: int,
) -> List[Tuple[str, str, int, int, int]]:
    """Generate the full list of experiment parameter tuples.

    Args:
        policy_names: Policies to benchmark.
        workload_names: Workloads to use.
        cache_sizes: Cache sizes to test.
        n_runs: Number of runs per configuration.
        base_seed: Base random seed.

    Returns:
        List of (policy_name, workload_name, cache_size, run_id, seed) tuples.
    """
    tasks: List[Tuple[str, str, int, int, int]] = []

    for workload_name in workload_names:
        for cache_size in cache_sizes:
            for run_id in range(n_runs):
                seed = base_seed + run_id
                for policy_name in policy_names:
                    tasks.append((policy_name, workload_name, cache_size, run_id, seed))

    return tasks


def run_parallel(
    tasks: List[Tuple[str, str, int, int, int]],
    n_workers: Optional[int] = None,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    """Execute experiment tasks in parallel using multiprocessing.

    Args:
        tasks: List of experiment parameter tuples.
        n_workers: Number of worker processes (default: CPU count - 1).
        show_progress: Whether to show a progress bar.

    Returns:
        List of result dictionaries.
    """
    if n_workers is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)

    n_workers = min(n_workers, len(tasks))

    print(f"Running {len(tasks)} experiments with {n_workers} workers...")

    if n_workers == 1:
        # Sequential execution (useful for debugging)
        results: List[Dict[str, Any]] = []
        iterator = tasks
        if show_progress and HAS_TQDM:
            iterator = tqdm(tasks, desc="Benchmarking", unit="exp")

        for task in iterator:
            result = _run_experiment_task(task)
            results.append(result)

        return results

    # Parallel execution
    with multiprocessing.Pool(processes=n_workers) as pool:
        if show_progress and HAS_TQDM:
            results = list(
                tqdm(
                    pool.imap_unordered(_run_experiment_task, tasks),
                    total=len(tasks),
                    desc="Benchmarking",
                    unit="exp",
                )
            )
        else:
            results = pool.map(_run_experiment_task, tasks)

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print a summary table of benchmark results.

    Groups by policy and workload, showing mean hit rate and cost-weighted
    hit rate across all runs and cache sizes.

    Args:
        results: List of result dictionaries.
    """
    if HAS_PANDAS:
        df = pd.DataFrame(results)
        summary = df.groupby(["policy_name", "workload_name"]).agg(
            mean_hit_rate=("hit_rate", "mean"),
            mean_cwhr=("cost_weighted_hit_rate", "mean"),
            mean_savings=("dollar_savings", "mean"),
            mean_throughput=("throughput", "mean"),
        ).round(4)
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(summary.to_string())
        print("=" * 80)
    else:
        # Manual summary without pandas
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY (install pandas for better formatting)")
        print("=" * 80)

        # Group by policy
        from collections import defaultdict
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in results:
            key = f"{r['policy_name']:10s} | {r['workload_name']:20s}"
            grouped[key].append(r)

        print(f"{'Policy':<10s} | {'Workload':<20s} | {'Hit Rate':>10s} | {'CWHR':>10s} | {'Savings':>10s}")
        print("-" * 70)
        for key, group in sorted(grouped.items()):
            hr = sum(r["hit_rate"] for r in group) / len(group)
            cwhr = sum(r["cost_weighted_hit_rate"] for r in group) / len(group)
            savings = sum(r["dollar_savings"] for r in group) / len(group)
            print(f"{key} | {hr:>10.4f} | {cwhr:>10.4f} | ${savings:>9.4f}")

        print("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run cache eviction policy benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick sanity check
  python -m benchmarks.run_all --quick

  # Full benchmark suite (30 runs, all configs)
  python -m benchmarks.run_all --n-runs 30 --output-dir results/

  # Specific policies and workloads
  python -m benchmarks.run_all --policies LRU GDSF --workloads adversarial_lru

  # Custom cache sizes
  python -m benchmarks.run_all --cache-sizes 256 512 1024 2048 4096
        """,
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=30,
        help="Number of repetitions per configuration (default: 30)",
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=list(WORKLOAD_REGISTRY.keys()),
        default=None,
        help="Workloads to run (default: all)",
    )
    parser.add_argument(
        "--cache-sizes",
        nargs="+",
        type=int,
        default=[50000, 100000, 250000, 500000],
        help="Cache sizes in bytes to test (default: 50000 100000 250000 500000)",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        choices=list(POLICY_REGISTRY.keys()),
        default=None,
        help="Policies to benchmark (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results (default: results/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count - 1)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quick sanity-check benchmark (smaller workloads, 1 run)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Force sequential execution (no multiprocessing)",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point for the benchmark CLI."""
    args = parse_args()

    print("=" * 80)
    print("  Cost-Aware Cache Eviction Policy Benchmark Suite")
    print("=" * 80)
    print()

    start_time = time.time()

    if args.quick:
        print("Running quick benchmark (sanity check)...")
        print("-" * 60)
        results_df = run_quick_benchmark(cache_size=100000, n_queries=5000, verbose=True)
        if HAS_PANDAS and hasattr(results_df, "to_dict"):
            results = results_df.to_dict("records")
        else:
            results = results_df
    else:
        # Full benchmark
        policy_names = args.policies or list(POLICY_REGISTRY.keys())
        workload_names = args.workloads or list(WORKLOAD_REGISTRY.keys())

        tasks = build_experiment_tasks(
            policy_names=policy_names,
            workload_names=workload_names,
            cache_sizes=args.cache_sizes,
            n_runs=args.n_runs,
            base_seed=args.seed,
        )

        n_workers = 1 if args.sequential else args.workers
        show_progress = not args.no_progress

        results = run_parallel(
            tasks=tasks,
            n_workers=n_workers,
            show_progress=show_progress,
        )

    elapsed = time.time() - start_time
    print(f"\nTotal benchmark time: {elapsed:.1f} seconds")

    # Print summary
    print_summary(results)

    # Save results
    save_results(results, output_dir=args.output_dir)

    # Also save run configuration
    config = {
        "n_runs": args.n_runs if not args.quick else 1,
        "cache_sizes": args.cache_sizes if not args.quick else [100000],
        "policies": args.policies or list(POLICY_REGISTRY.keys()),
        "workloads": args.workloads or list(WORKLOAD_REGISTRY.keys()),
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    config_path = Path(args.output_dir) / "benchmark_config.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Config: {config_path}")


if __name__ == "__main__":
    main()
