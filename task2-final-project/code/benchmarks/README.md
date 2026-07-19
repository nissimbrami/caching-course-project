# Benchmark Suite for Cost-Aware Cache Eviction (GDSF)

This benchmark suite evaluates the GDSF (Greedy Dual-Size Frequency) eviction policy against standard baselines (LRU, FIFO, LFU, Random) across multiple workloads designed to stress-test cost-aware eviction.

## Quick Start

Run a quick sanity check (takes ~10 seconds):

```bash
python -m benchmarks.run_all --quick
```

## Full Benchmark Suite

Run the complete benchmark with 30 repetitions per configuration (for statistical significance):

```bash
python -m benchmarks.run_all --n-runs 30 --output-dir results/
```

## Custom Configurations

### Select specific policies:
```bash
python -m benchmarks.run_all --policies LRU GDSF --n-runs 30
```

### Select specific workloads:
```bash
python -m benchmarks.run_all --workloads adversarial_lru high_variance_cost
```

### Custom cache sizes:
```bash
python -m benchmarks.run_all --cache-sizes 256 512 1024 2048 4096
```

### Sequential execution (for debugging):
```bash
python -m benchmarks.run_all --sequential --n-runs 1
```

### Control parallelism:
```bash
python -m benchmarks.run_all --workers 4
```

## Workloads

| Workload | Description | What it tests |
|----------|-------------|---------------|
| `uniform_cost` | All queries have the same cost | Baseline (cost provides no signal) |
| `high_variance_cost` | Trimodal: 60% cheap, 30% medium, 10% expensive | Cost-aware retention of expensive items |
| `zipf_variable_cost` | Zipfian popularity, rare items are expensive | Trade-off between frequency and cost |
| `bursty` | Temporal bursts of related queries | Adaptability to changing access patterns |
| `adversarial_lru` | Scan pattern that pollutes LRU + expensive recurring queries | GDSF advantage over LRU |
| `size_varying` | Response sizes from 50B to 50KB | Size-awareness (the S in GDSF) |

## Output Format

Results are saved as both CSV and JSON:

### CSV columns:
- `policy_name` - Name of the eviction policy
- `workload_name` - Name of the workload generator
- `cache_size` - Cache capacity in bytes
- `hit_rate` - Fraction of queries served from cache
- `cost_weighted_hit_rate` - Sum of hit costs / sum of all costs
- `dollar_savings` - Total dollar cost saved by cache hits
- `latency_p50` - Median operation latency (ms)
- `latency_p95` - 95th percentile latency (ms)
- `latency_p99` - 99th percentile latency (ms)
- `throughput` - Queries processed per second
- `memory_overhead_bytes` - Memory used by the policy
- `n_queries` - Total queries in the workload
- `n_evictions` - Number of evictions performed
- `run_id` - Repetition identifier (0 to n_runs-1)
- `seed` - Random seed for this run

### JSON output:
Same fields as CSV but in JSON array format for programmatic consumption.

### Configuration file:
A `benchmark_config.json` file is also saved with the run parameters.

## Expected Runtime Estimates

| Configuration | Approximate Runtime |
|---------------|-------------------|
| `--quick` | ~10 seconds |
| Default (30 runs, all configs) | ~15-30 minutes |
| Single workload, 2 policies, 30 runs | ~2-3 minutes |
| Full suite, 4 workers | ~8-15 minutes |

Runtime scales linearly with: n_policies x n_workloads x n_cache_sizes x n_runs

## Dependencies

Required:
- `numpy` (workload generation)

Optional (recommended):
- `pandas` (result aggregation and export)
- `tqdm` (progress bars)

Install all:
```bash
pip install numpy pandas tqdm
```

## Reproducibility

All experiments are fully deterministic given the same seed. The default base seed is 42, and each run uses `base_seed + run_id` for its random state. To reproduce results:

```bash
python -m benchmarks.run_all --seed 42 --n-runs 30 --sequential
```

The `--sequential` flag ensures deterministic execution order (parallel execution may reorder results but each individual result is still reproducible).
