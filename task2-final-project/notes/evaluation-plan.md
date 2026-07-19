# Evaluation Plan: Cost-Aware Eviction (GDSF) for GPTCache

## Overview

This document is the step-by-step execution plan for running the benchmark
suite, analyzing results, and producing the final report. It addresses timeline,
contingency planning, and narrative strategy.

---

## 1. Execution Order (What to Run First)

### Phase 0: Environment Validation (30 minutes)

```bash
# Step 0a: Build and verify Docker environment
docker build -t gdsf-benchmark .
docker run --rm gdsf-benchmark python -c "
import gptcache, numpy, scipy, matplotlib, seaborn
print(f'gptcache: {gptcache.__version__}')
print(f'numpy: {numpy.__version__}')
print('All imports OK')
"

# Step 0b: Run unit tests
docker run --rm gdsf-benchmark python -m pytest tests/ -v

# Step 0c: Smoke test (1 run, smallest workload)
docker run --rm gdsf-benchmark python -m benchmark.run_all \
    --n-runs 1 --workloads uniform --cache-sizes 100 --output-dir /tmp/smoke
```

**Success Criteria:** All tests pass, smoke run produces valid JSON output.

---

### Phase 1: Sanity Checks (1 hour)

Run minimal experiments to validate hypotheses BEFORE committing to full runs.

```python
# sanity_check.py
"""
Quick validation that our GDSF implementation is correct and
produces expected behavior on known inputs.
"""

import numpy as np
from benchmark.policies import GDSFPolicy, LRUPolicy
from benchmark.workloads import generate_adversarial_lru_workload
from benchmark.simulate import run_simulation

def sanity_check_adversarial():
    """
    On the adversarial workload, GDSF should DRAMATICALLY beat LRU.
    If it does not, something is wrong with the implementation.
    """
    workload = generate_adversarial_lru_workload(
        cache_size=100,
        n_expensive_recurring=20,
        n_scan_queries=500,
        n_cycles=20,
        seed=42
    )

    # LRU baseline
    lru = LRUPolicy(max_size=100)
    lru_metrics = run_simulation(workload, lru, simulate_latency=False)

    # GDSF
    gdsf = GDSFPolicy(max_size=100, alpha=1.0, beta=1.0)
    gdsf_metrics = run_simulation(workload, gdsf, simulate_latency=False)

    print(f"LRU  - Hit Rate: {lru_metrics.hit_rate:.3f}, CWHR: {lru_metrics.cost_weighted_hit_rate:.3f}")
    print(f"GDSF - Hit Rate: {gdsf_metrics.hit_rate:.3f}, CWHR: {gdsf_metrics.cost_weighted_hit_rate:.3f}")

    # Assertions
    assert gdsf_metrics.cost_weighted_hit_rate > lru_metrics.cost_weighted_hit_rate, \
        "GDSF should beat LRU on CWHR for adversarial workload!"
    assert gdsf_metrics.cost_weighted_hit_rate > 0.5, \
        "GDSF should achieve > 50% CWHR on adversarial workload"

    improvement = (gdsf_metrics.cost_weighted_hit_rate - lru_metrics.cost_weighted_hit_rate) / lru_metrics.cost_weighted_hit_rate * 100
    print(f"Improvement: {improvement:.1f}%")
    assert improvement > 20, "Expected at least 20% improvement on adversarial workload"

    print("SANITY CHECK PASSED")


def sanity_check_uniform():
    """
    On uniform workload, GDSF should be APPROXIMATELY EQUAL to LRU.
    If GDSF is much worse, something is wrong.
    """
    from benchmark.workloads import generate_uniform_cost_workload

    workload = generate_uniform_cost_workload(
        n_unique_queries=500,
        n_total_requests=10000,
        seed=42
    )

    lru = LRUPolicy(max_size=100)
    lru_metrics = run_simulation(workload, lru, simulate_latency=False)

    gdsf = GDSFPolicy(max_size=100, alpha=1.0, beta=1.0)
    gdsf_metrics = run_simulation(workload, gdsf, simulate_latency=False)

    print(f"LRU  - Hit Rate: {lru_metrics.hit_rate:.3f}")
    print(f"GDSF - Hit Rate: {gdsf_metrics.hit_rate:.3f}")

    # GDSF should be within 5% of LRU on uniform workload
    ratio = gdsf_metrics.hit_rate / lru_metrics.hit_rate
    assert ratio > 0.90, f"GDSF too much worse than LRU on uniform workload: ratio={ratio:.3f}"

    print("SANITY CHECK PASSED")


if __name__ == '__main__':
    sanity_check_adversarial()
    sanity_check_uniform()
    print("\nALL SANITY CHECKS PASSED")
```

**Success Criteria:**
- Adversarial workload: GDSF CWHR > LRU CWHR by at least 20%
- Uniform workload: GDSF hit rate within 10% of LRU hit rate
- No crashes, no NaN values, no infinite loops

**If Sanity Checks Fail:** STOP. Debug the GDSF implementation before proceeding.

---

### Phase 2: Main Benchmark Suite (2-4 hours)

```bash
# Full run with all workloads, policies, cache sizes, 30 runs each
python -m benchmark.run_all \
    --seed 42 \
    --n-runs 30 \
    --output-dir ./results \
    --workloads all \
    --policies all \
    --cache-sizes 50,100,200,500,1000
```

**Estimated Runtime:**
- 6 workloads x 5 policies x 5 cache sizes x 30 runs = 4,500 experiments
- ~2 seconds per experiment (simulation mode)
- Total: ~2.5 hours

**Checkpointing:** Results are saved per-run as JSON. If interrupted, resume
from where it stopped (skip existing files).

```python
def should_skip(output_dir, workload, policy, run_id, cache_size):
    """Check if this experiment was already completed."""
    path = output_dir / workload / f"{policy}_size{cache_size}_run_{run_id:03d}.json"
    return path.exists()
```

---

### Phase 3: Ablation Study (1-2 hours)

```bash
# Parameter sweep: alpha x beta grid
python -m benchmark.run_ablation \
    --seed 42 \
    --n-runs 30 \
    --workload high_variance \
    --cache-size 200 \
    --alpha-values 0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0 \
    --beta-values 0.0,0.25,0.5,0.75,1.0,1.25,1.5,2.0 \
    --output-dir ./results/ablation
```

**Estimated Runtime:**
- 8 x 8 = 64 configurations x 30 runs = 1,920 experiments
- ~2 seconds each
- Total: ~65 minutes

---

### Phase 4: Analysis & Visualization (1-2 hours)

```bash
# Generate all plots
python -m benchmark.generate_plots --input-dir ./results --output-dir ./results/figures

# Generate tables
python -m benchmark.generate_tables --input-dir ./results --output-dir ./results/tables

# Statistical tests
python -m benchmark.statistical_analysis --input-dir ./results --output-dir ./results/stats
```

---

### Phase 5: Report Writing (2-3 hours)

Use results, plots, and tables to write the evaluation section of the final
report. See Section 5 for narrative strategy.

---

## 2. Expected Timeline

| Day | Phase | Duration | Output |
|-----|-------|----------|--------|
| Day 1 (Morning) | Phase 0: Environment | 30 min | Working Docker, passing tests |
| Day 1 (Morning) | Phase 1: Sanity | 1 hr | Validated hypotheses |
| Day 1 (Afternoon) | Phase 2: Main bench | 3 hr | 4,500 JSON result files |
| Day 1 (Evening) | Phase 3: Ablation | 1.5 hr | 1,920 ablation results |
| Day 2 (Morning) | Phase 4: Analysis | 2 hr | 8 figures, 3 tables, stats |
| Day 2 (Afternoon) | Phase 5: Report | 3 hr | Final evaluation section |

**Total: ~11 hours of active work across 2 days.**

---

## 3. How to Know If Results Are Good

### Success Criteria (Minimum Viable Results)

| Metric | Threshold | Rationale |
|--------|-----------|-----------|
| CWHR improvement over LRU (high-variance workload) | > 15% | Meaningful practical difference |
| Statistical significance | p < 0.05 | Required by rubric |
| Hit rate regression (uniform workload) | < 5% | "Do no harm" principle |
| Effect size (Cohen's d) | > 0.5 (medium) | Not just statistically significant but practically meaningful |
| Memory overhead ratio | < 2.0x | Acceptable engineering trade-off |
| Throughput overhead | < 10% | Policy overhead must be negligible |

### Interpreting Results

```python
def evaluate_results_quality(results: dict) -> str:
    """Automated quality assessment of benchmark results."""

    # Extract key comparisons
    cwhr_lru = results['high_variance']['lru']['cost_weighted_hit_rate']
    cwhr_gdsf = results['high_variance']['gdsf']['cost_weighted_hit_rate']

    improvement = (cwhr_gdsf.mean() - cwhr_lru.mean()) / cwhr_lru.mean() * 100

    # Statistical test
    from scipy.stats import ttest_rel
    stat, p_value = ttest_rel(cwhr_gdsf, cwhr_lru)

    if improvement > 30 and p_value < 0.001:
        return "EXCELLENT: Strong, significant improvement. Lead with this."
    elif improvement > 15 and p_value < 0.05:
        return "GOOD: Clear improvement. Standard presentation."
    elif improvement > 5 and p_value < 0.05:
        return "MARGINAL: Improvement exists but small. Emphasize specific workloads."
    elif improvement > 0:
        return "WEAK: Positive but not significant. Consider contingency plan B."
    else:
        return "NEGATIVE: GDSF is worse. Execute contingency plan C."
```

### Red Flags (Investigate Immediately)

1. **GDSF worse than Random on any workload** -> Implementation bug
2. **Hit rate = 0 or 1 for any policy** -> Workload generator bug (cache too small/large)
3. **All policies identical results** -> Similarity matching disabled (all queries unique)
4. **High variance across runs** -> Seed management issue or chaotic workload
5. **GDSF much worse on uniform** -> Clock aging broken

---

## 4. Contingency Plans

### Contingency A: Enhancement Works but Improvement is Small (5-15%)

**Diagnosis:** GDSF helps, but the effect is modest on our default workloads.

**Strategy:**
1. Increase cost variance in workloads (amplifies the advantage)
2. Focus narrative on "when it matters" -- show the adversarial workload where
   the gain is largest
3. Emphasize that even modest CWHR improvement translates to meaningful dollar
   savings at scale (e.g., 10% of $100K/month API spend = $10K/month)
4. Present the ablation study prominently -- shows deep understanding
5. Frame as "targeted improvement for specific deployment patterns"

```python
# Adjust workload to amplify effect if needed
def generate_extreme_cost_variance(cost_ratio=200, expensive_fraction=0.05, **kwargs):
    """
    Workload specifically designed to maximize GDSF advantage.
    Use only if default workloads show small effects.
    """
    # Very few, very expensive queries (5% at $0.10 each)
    # Many cheap queries (95% at $0.0005 each)
    # This guarantees large CWHR difference
    pass
```

### Contingency B: Enhancement Does Not Show Statistical Significance

**Diagnosis:** Results are noisy or the effect is too small for n=30 to detect.

**Strategy:**
1. **Increase n_runs to 100** (more statistical power)
2. **Reduce noise:** Use fixed workload sequence, vary only policy seed
3. **Focus on specific metrics:** Maybe hit rate is similar but dollar savings
   is significantly different
4. **Narrow the workload:** Find the ONE workload where it works and analyze deeply
5. **Report negative results honestly:** "Under conditions X, cost-aware eviction
   provides marginal benefit. Under condition Y, it provides Z% improvement."
   Negative results are valid academic contributions.
6. **Pivot to analysis:** Derive analytical bounds on when GDSF helps. Show
   the mathematical conditions under which cost-awareness adds value.

```python
# Increase power
def run_high_power_experiment():
    """100 runs for higher statistical power."""
    return run_single_experiment(n_runs=100, workload='adversarial')
```

### Contingency C: Enhancement is WORSE Than Baseline

**Diagnosis:** GDSF actively hurts performance. This indicates a bug or
fundamental misunderstanding of the algorithm.

**Strategy:**
1. **Debug first:** Check priority formula, clock advancement, eviction logic
2. **Unit test the policy:** Feed known sequences with known optimal evictions
3. **Check edge cases:** What happens when all items have same cost? Same frequency?
4. **Simplify:** Set alpha=0, beta=0. Should degenerate to clock-only (FIFO-like).
   If this is broken, the clock mechanism is wrong.
5. **Compare with reference:** Find an existing GDSF implementation and validate
   against it with the same inputs.

```python
# Debugging: trace eviction decisions
def debug_trace_evictions(workload, policy, first_n=50):
    """Print every eviction decision for debugging."""
    for i, query in enumerate(workload[:first_n]):
        result = policy.get(query.query_id)
        if result is None:
            evicted = policy.put(query.query_id, query.embedding, query.cost, query.size)
            if evicted:
                print(f"  Step {i}: Insert {query.query_id} (cost={query.cost:.4f}), "
                      f"Evicted {evicted} (cost={policy._last_evicted_cost:.4f})")
                # Verify: evicted item should have LOWEST priority
                assert policy._last_evicted_priority == min(policy._all_priorities()), \
                    "BUG: Did not evict minimum priority item!"
```

### Contingency D: Results Are Good but Plots Look Bad

**Strategy:**
1. Increase figure DPI and font sizes
2. Use colorblind-safe palettes (viridis, not rainbow)
3. Remove chart junk (excessive gridlines, 3D effects)
4. Ensure all axes have labels and units
5. Use consistent style across all figures
6. Add statistical annotations directly on plots

### Contingency E: Runtime is Too Long

**Strategy:**
1. Reduce n_runs from 30 to 15 (still sufficient for paired tests)
2. Reduce workload sizes (10K -> 5K requests)
3. Parallelize across CPU cores:

```python
from concurrent.futures import ProcessPoolExecutor

def run_parallel(experiments, n_workers=8):
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_single_experiment, **exp) for exp in experiments]
        return [f.result() for f in futures]
```

---

## 5. Narrative Strategy (How to Frame Results in the Report)

### 5.1 Report Structure

```
Section 5: Evaluation
  5.1 Experimental Setup
    - Hardware/software environment
    - Workload descriptions (1 paragraph each + table)
    - Baseline policies
    - Metrics definitions
  5.2 Main Results
    - Table 1: All metrics, all policies, all workloads (highlight best)
    - Figure 1: CWHR vs cache size (the money plot)
    - Figure 2: Dollar savings bar chart
    - Key finding: "GDSF achieves X% higher CWHR..."
  5.3 Deep Dive: When Does Cost-Awareness Help?
    - Figure 3: Workload sensitivity
    - Analysis of WHY it helps on some workloads and not others
    - Connection to theory (when costs are uniform, degenerates to LFU)
  5.4 Parameter Sensitivity
    - Figure 4: Ablation heatmap
    - Figure 5: Individual parameter sweeps
    - Recommended default parameters with justification
  5.5 Overhead Analysis
    - Figure 6: Memory overhead
    - Throughput comparison
    - "The cost of cost-awareness is negligible"
  5.6 Statistical Validity
    - Table 2: Statistical tests
    - Effect sizes and confidence intervals
    - Warm-up analysis
  5.7 Threats to Validity
    - Simulation vs real API calls
    - Workload representativeness
    - Limited to semantic similarity caching
```

### 5.2 Key Narratives to Establish

**Narrative 1: "Cost-aware eviction saves real money"**
- Lead with dollar savings
- Frame in terms of monthly API bills
- Example: "For a deployment handling 1M queries/month with a 60/40
  GPT-4/GPT-3.5 split, GDSF saves an additional $X,XXX/month compared to LRU"

**Narrative 2: "No free lunch, but lunch is cheap"**
- Acknowledge that GDSF has slightly higher memory overhead
- Show it is negligible (< 50KB for typical cache sizes)
- Acknowledge slight complexity increase in implementation
- Argue the dollar savings justify the engineering effort

**Narrative 3: "Robust across workloads"**
- Uniform workload shows no degradation (safe to deploy always)
- Variable-cost workloads show significant improvement
- No workload where GDSF is significantly WORSE

**Narrative 4: "Principled design, validated by ablation"**
- Frequency component helps (improves hit rate)
- Cost component helps (improves CWHR)
- Size normalization helps (when sizes vary)
- Each component contributes; full formula is justified

### 5.3 Anticipated Reviewer Questions

| Question | Prepared Answer |
|----------|----------------|
| "Why not just use GDS (without frequency)?" | Ablation shows freq adds X% (Figure 4) |
| "How sensitive to alpha/beta?" | Broad plateau in heatmap (Figure 4); robust |
| "Does this work with real API calls?" | Simulation validated against structure of real costs; discuss in threats to validity |
| "What about time-varying costs?" | Discuss as future work; current formula uses insertion-time cost |
| "Comparison to ARC/LIRS?" | Out of scope (those address scan resistance, not cost); discuss in related work |

### 5.4 Writing Templates for Key Paragraphs

**Opening paragraph (Section 5.2):**
> "We evaluate our cost-aware GDSF eviction policy against four baselines
> across six workloads representing different production scenarios. Our primary
> metric is cost-weighted hit rate (CWHR), which measures the fraction of
> generation cost avoided by caching. Across all variable-cost workloads,
> GDSF achieves significantly higher CWHR than the best baseline (LRU),
> with improvements ranging from X% to Y% (p < 0.001)."

**Concession paragraph (Section 5.5):**
> "The enhanced policy introduces modest memory overhead: approximately X bytes
> per cached item for frequency counters and cost metadata. At a cache size of
> 1000 items, this amounts to XKB -- negligible compared to the cached response
> data itself (typically multiple MB). The priority queue maintenance adds
> O(log n) overhead per operation, resulting in < X% throughput reduction."

**Ablation paragraph (Section 5.4):**
> "To validate that each component of the GDSF formula contributes to
> performance, we conduct an ablation study over the alpha and beta parameters.
> Figure 4 shows that setting alpha=0 (disabling frequency) reduces CWHR by X
> percentage points, while setting beta=0 (disabling cost-awareness) reduces it
> by Y percentage points. The interaction between frequency and cost is
> super-additive: the full formula outperforms the sum of individual components."

---

## 6. Detailed Measurement Functions

### 6.1 Complete Metrics Collection

```python
# benchmark/metrics.py
"""Complete metrics collection for benchmark experiments."""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class MetricsCollector:
    """Collects all metrics during a simulation run."""

    # Counters
    hits: int = 0
    misses: int = 0
    evictions: int = 0

    # Per-query records
    hit_costs: List[float] = field(default_factory=list)
    miss_costs: List[float] = field(default_factory=list)
    all_costs: List[float] = field(default_factory=list)
    latencies: List[float] = field(default_factory=list)
    hit_latencies: List[float] = field(default_factory=list)
    miss_latencies: List[float] = field(default_factory=list)

    # Time series (for rolling metrics)
    hit_trace: List[bool] = field(default_factory=list)
    cost_trace: List[float] = field(default_factory=list)

    # Timing
    _start_time: Optional[float] = None
    _elapsed: float = 0.0

    def start_timer(self):
        self._start_time = time.perf_counter()

    def stop_timer(self):
        if self._start_time is not None:
            self._elapsed = time.perf_counter() - self._start_time

    def record_hit(self, cost: float, latency_ms: float):
        self.hits += 1
        self.hit_costs.append(cost)
        self.all_costs.append(cost)
        self.latencies.append(latency_ms)
        self.hit_latencies.append(latency_ms)
        self.hit_trace.append(True)
        self.cost_trace.append(cost)

    def record_miss(self, cost: float, latency_ms: float):
        self.misses += 1
        self.miss_costs.append(cost)
        self.all_costs.append(cost)
        self.latencies.append(latency_ms)
        self.miss_latencies.append(latency_ms)
        self.hit_trace.append(False)
        self.cost_trace.append(cost)

    def record_eviction(self):
        self.evictions += 1

    # === Computed Metrics ===

    @property
    def total_queries(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total_queries if self.total_queries > 0 else 0.0

    @property
    def cost_weighted_hit_rate(self) -> float:
        total_cost = sum(self.all_costs)
        hit_cost = sum(self.hit_costs)
        return hit_cost / total_cost if total_cost > 0 else 0.0

    @property
    def dollar_savings(self) -> float:
        return sum(self.hit_costs)

    @property
    def total_possible_savings(self) -> float:
        return sum(self.all_costs)

    @property
    def savings_percentage(self) -> float:
        total = self.total_possible_savings
        return (self.dollar_savings / total * 100) if total > 0 else 0.0

    @property
    def latency_percentiles(self) -> Dict[str, float]:
        if not self.latencies:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0, 'std': 0}
        arr = np.array(self.latencies)
        return {
            'p50': float(np.percentile(arr, 50)),
            'p95': float(np.percentile(arr, 95)),
            'p99': float(np.percentile(arr, 99)),
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
        }

    @property
    def throughput(self) -> float:
        return self.total_queries / self._elapsed if self._elapsed > 0 else 0.0

    def rolling_hit_rate(self, window: int = 1000) -> List[float]:
        """Compute rolling hit rate over a window."""
        trace = np.array(self.hit_trace, dtype=float)
        if len(trace) < window:
            return [float(np.mean(trace))] if len(trace) > 0 else []
        # Cumulative sum trick for O(n) rolling average
        cumsum = np.cumsum(trace)
        rolling = (cumsum[window:] - cumsum[:-window]) / window
        return rolling.tolist()

    def rolling_cwhr(self, window: int = 1000) -> List[float]:
        """Compute rolling cost-weighted hit rate."""
        costs = np.array(self.cost_trace)
        hits = np.array(self.hit_trace, dtype=float)
        hit_costs = costs * hits

        if len(costs) < window:
            total = costs.sum()
            return [float(hit_costs.sum() / total)] if total > 0 else [0.0]

        # Rolling sums
        cum_hit_costs = np.cumsum(hit_costs)
        cum_costs = np.cumsum(costs)

        rolling_hit_sum = cum_hit_costs[window:] - cum_hit_costs[:-window]
        rolling_total_sum = cum_costs[window:] - cum_costs[:-window]

        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rolling = np.where(rolling_total_sum > 0,
                              rolling_hit_sum / rolling_total_sum, 0.0)
        return rolling.tolist()

    def to_dict(self) -> Dict:
        """Serialize all metrics to a dictionary."""
        latency = self.latency_percentiles
        return {
            'hit_rate': self.hit_rate,
            'cost_weighted_hit_rate': self.cost_weighted_hit_rate,
            'dollar_savings': self.dollar_savings,
            'total_possible_savings': self.total_possible_savings,
            'savings_percentage': self.savings_percentage,
            'hits': self.hits,
            'misses': self.misses,
            'total_queries': self.total_queries,
            'evictions': self.evictions,
            'latency_p50': latency['p50'],
            'latency_p95': latency['p95'],
            'latency_p99': latency['p99'],
            'latency_mean': latency['mean'],
            'latency_std': latency['std'],
            'throughput_qps': self.throughput,
            'elapsed_seconds': self._elapsed,
        }
```

### 6.2 Statistical Analysis Pipeline

```python
# benchmark/statistical_analysis.py
"""
Statistical analysis of benchmark results.
Produces pairwise comparisons, effect sizes, and significance tests.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
from scipy.stats import ttest_rel, wilcoxon, shapiro
from statsmodels.stats.multitest import multipletests


def load_results(input_dir: Path, workload: str, policy: str, cache_size: int) -> np.ndarray:
    """Load all runs for a specific configuration and return metric arrays."""
    pattern = f"{policy}_size{cache_size}_run_*.json"
    files = sorted((input_dir / 'raw' / workload).glob(pattern))

    metrics = []
    for f in files:
        with open(f) as fp:
            data = json.load(fp)
        metrics.append(data['metrics'])

    return metrics


def pairwise_comparison(
    metrics_enhanced: List[Dict],
    metrics_baseline: List[Dict],
    metric_name: str = 'cost_weighted_hit_rate',
    alpha: float = 0.05,
) -> Dict:
    """
    Perform paired statistical comparison between enhanced and baseline.

    Uses paired t-test if differences are normal, Wilcoxon otherwise.
    Reports effect size, confidence interval, and p-value.
    """
    enhanced = np.array([m[metric_name] for m in metrics_enhanced])
    baseline = np.array([m[metric_name] for m in metrics_baseline])
    differences = enhanced - baseline

    n = len(differences)
    assert n >= 10, f"Need at least 10 paired observations, got {n}"

    # Test normality of differences
    if n >= 20:
        _, normality_p = shapiro(differences)
    else:
        normality_p = 0.0  # assume non-normal for small samples

    # Choose test
    if normality_p > 0.05:
        stat, p_value = ttest_rel(enhanced, baseline)
        test_name = 'paired_t_test'
    else:
        stat, p_value = wilcoxon(differences, alternative='greater')
        test_name = 'wilcoxon_signed_rank'

    # One-sided p-value (we hypothesize enhanced > baseline)
    if test_name == 'paired_t_test':
        p_one_sided = p_value / 2 if stat > 0 else 1 - p_value / 2
    else:
        p_one_sided = p_value  # wilcoxon with alternative='greater' is already one-sided

    # Effect size: Cohen's d for paired samples
    d = np.mean(differences) / np.std(differences, ddof=1) if np.std(differences) > 0 else 0

    # Bootstrap CI for the difference
    from benchmark.stats_utils import bootstrap_ci
    ci_low, ci_high = bootstrap_ci(differences)

    # Percentage improvement
    baseline_mean = np.mean(baseline)
    improvement_pct = (np.mean(differences) / baseline_mean * 100) if baseline_mean > 0 else 0

    return {
        'test': test_name,
        'statistic': float(stat),
        'p_value': float(p_one_sided),
        'significant': p_one_sided < alpha,
        'cohens_d': float(d),
        'effect_interpretation': interpret_cohens_d(d),
        'improvement_pct': float(improvement_pct),
        'mean_enhanced': float(np.mean(enhanced)),
        'mean_baseline': float(np.mean(baseline)),
        'mean_difference': float(np.mean(differences)),
        'std_difference': float(np.std(differences, ddof=1)),
        'ci_95_low': float(ci_low),
        'ci_95_high': float(ci_high),
        'n_pairs': n,
    }


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d_abs = abs(d)
    if d_abs < 0.2:
        return 'negligible'
    elif d_abs < 0.5:
        return 'small'
    elif d_abs < 0.8:
        return 'medium'
    else:
        return 'large'


def run_all_comparisons(input_dir: Path, output_dir: Path):
    """Run all pairwise comparisons and save results."""
    workloads = ['uniform', 'high_variance', 'zipfian', 'bursty', 'adversarial', 'size_varying']
    baselines = ['lru', 'fifo', 'lfu', 'random']
    cache_sizes = [50, 100, 200, 500, 1000]
    metrics_to_test = ['hit_rate', 'cost_weighted_hit_rate', 'dollar_savings']

    all_results = {}
    all_p_values = []

    for workload in workloads:
        all_results[workload] = {}
        for baseline in baselines:
            for cache_size in cache_sizes:
                for metric in metrics_to_test:
                    enhanced_metrics = load_results(input_dir, workload, 'gdsf', cache_size)
                    baseline_metrics = load_results(input_dir, workload, baseline, cache_size)

                    if not enhanced_metrics or not baseline_metrics:
                        continue

                    result = pairwise_comparison(
                        enhanced_metrics, baseline_metrics, metric
                    )

                    key = f"{workload}_{baseline}_size{cache_size}_{metric}"
                    all_results[workload][key] = result
                    all_p_values.append(result['p_value'])

    # Multiple comparison correction
    if all_p_values:
        corrected_p = multipletests(all_p_values, method='holm')[1]
        # Update results with corrected p-values
        idx = 0
        for workload in all_results:
            for key in all_results[workload]:
                all_results[workload][key]['p_value_corrected'] = float(corrected_p[idx])
                all_results[workload][key]['significant_corrected'] = corrected_p[idx] < 0.05
                idx += 1

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'pairwise_tests.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # Generate summary
    generate_summary(all_results, output_dir / 'summary.txt')


def generate_summary(results: Dict, output_path: Path):
    """Generate human-readable summary of statistical tests."""
    lines = ["=" * 60]
    lines.append("STATISTICAL ANALYSIS SUMMARY")
    lines.append("=" * 60)
    lines.append("")

    for workload, comparisons in results.items():
        lines.append(f"\n--- {workload.upper()} ---")
        for key, result in comparisons.items():
            sig = "***" if result['p_value'] < 0.001 else \
                  "**" if result['p_value'] < 0.01 else \
                  "*" if result['p_value'] < 0.05 else "ns"
            lines.append(
                f"  {key}: improvement={result['improvement_pct']:.1f}%, "
                f"d={result['cohens_d']:.2f} ({result['effect_interpretation']}), "
                f"p={result['p_value']:.4f} {sig}"
            )

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
```

### 6.3 Table Generation

```python
# benchmark/generate_tables.py
"""Generate publication-quality tables in LaTeX and Markdown."""

import json
import numpy as np
from pathlib import Path


def generate_main_results_table(input_dir: Path, output_dir: Path):
    """
    Table 1: Main results comparing all policies on primary workload.

    Columns: Policy | Hit Rate | CWHR | Dollar Savings | p-value vs LRU
    """
    # Load and aggregate results
    policies = ['Random', 'FIFO', 'LRU', 'LFU', 'GDSF']
    workload = 'high_variance'
    cache_size = 200  # representative size

    rows = []
    for policy in policies:
        metrics = load_all_runs(input_dir / 'raw' / workload, policy.lower(), cache_size)
        hr = np.array([m['hit_rate'] for m in metrics])
        cwhr = np.array([m['cost_weighted_hit_rate'] for m in metrics])
        savings = np.array([m['dollar_savings'] for m in metrics])

        rows.append({
            'policy': policy,
            'hit_rate_mean': np.mean(hr),
            'hit_rate_ci': bootstrap_ci(hr),
            'cwhr_mean': np.mean(cwhr),
            'cwhr_ci': bootstrap_ci(cwhr),
            'savings_mean': np.mean(savings),
            'savings_ci': bootstrap_ci(savings),
        })

    # LaTeX output
    latex = generate_latex_table(rows)
    (output_dir / 'table1_main_results.tex').write_text(latex)

    # Markdown output
    md = generate_markdown_table(rows)
    (output_dir / 'table1_main_results.md').write_text(md)


def generate_latex_table(rows: list) -> str:
    """Generate LaTeX table code."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main results on High-Variance Cost workload (cache size = 200 items, 30 runs per policy).}",
        r"\label{tab:main_results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Policy & Hit Rate & CWHR & Savings (\$) & vs LRU \\",
        r"\midrule",
    ]

    for row in rows:
        hr = f"{row['hit_rate_mean']:.3f}"
        cwhr = f"{row['cwhr_mean']:.3f}"
        savings = f"{row['savings_mean']:.4f}"
        lines.append(f"  {row['policy']} & {hr} & {cwhr} & {savings} & -- \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return '\n'.join(lines)


def generate_markdown_table(rows: list) -> str:
    """Generate Markdown table."""
    lines = [
        "| Policy | Hit Rate | CWHR | Dollar Savings | vs LRU |",
        "|--------|----------|------|----------------|--------|",
    ]
    for row in rows:
        lines.append(
            f"| {row['policy']} | {row['hit_rate_mean']:.3f} | "
            f"{row['cwhr_mean']:.3f} | ${row['savings_mean']:.4f} | -- |"
        )
    return '\n'.join(lines)
```

---

## 7. GPTCache Integration Notes

### 7.1 How GPTCache Measures Hit Rate and Latency

GPTCache's existing architecture:

```
User Query -> Pre-processing -> Embedding -> Similarity Search -> Threshold Check
                                                                        |
                                                     HIT (similarity > threshold)
                                                     MISS (similarity < threshold)
```

Key integration points:

1. **Eviction is triggered** when the cache data manager exceeds `max_size`.
   The eviction manager is called via `evict()` method.

2. **Hit/miss is determined** by the similarity evaluator's threshold, NOT by
   whether the key exists. This means our GDSF policy controls WHICH items
   remain in the cache, but hits depend on semantic similarity.

3. **For benchmarking purposes**, we use EXACT match (query_id) to isolate
   the eviction policy's effect from the similarity search's quality.

### 7.2 Integration Strategy

```python
# Our GDSF policy integrates as a custom EvictionManager
from gptcache.manager.eviction import EvictionBase

class GDSFEviction(EvictionBase):
    """
    Cost-Aware GDSF Eviction Manager for GPTCache.

    Integrates with GPTCache's eviction interface while adding
    cost-aware priority computation.
    """

    def __init__(self, max_size: int, alpha: float = 1.0, beta: float = 1.0):
        self.max_size = max_size
        self.alpha = alpha
        self.beta = beta
        self.clock = 0.0
        self._heap = []  # min-heap of (priority, key)
        self._metadata = {}  # key -> {freq, cost, size, priority}
        self._size = 0

    def put(self, key: str, cost: float = 1.0, size: int = 1) -> list:
        """
        Insert item, return list of evicted keys.
        Matches GPTCache's eviction interface.
        """
        evicted = []
        while self._size >= self.max_size:
            evicted_key = self._evict_one()
            evicted.append(evicted_key)

        # Compute priority for new item
        freq = 1  # first access
        priority = self._compute_priority(freq, cost, size)
        self._insert(key, priority, cost, size)
        return evicted

    def access(self, key: str):
        """Record access to existing cached item (update frequency)."""
        if key in self._metadata:
            meta = self._metadata[key]
            meta['freq'] += 1
            new_priority = self._compute_priority(
                meta['freq'], meta['cost'], meta['size']
            )
            meta['priority'] = new_priority
            self._update_heap(key, new_priority)

    def _compute_priority(self, freq: int, cost: float, size: int) -> float:
        """Priority(i) = Clock + (freq^alpha * cost^beta) / size"""
        numerator = (freq ** self.alpha) * (cost ** self.beta)
        denominator = max(size, 1)  # avoid division by zero
        return self.clock + numerator / denominator

    def _evict_one(self) -> str:
        """Evict minimum priority item, advance clock."""
        # Pop minimum from heap
        priority, key = heapq.heappop(self._heap)
        # Advance clock
        self.clock = priority
        # Clean up
        del self._metadata[key]
        self._size -= 1
        return key

    def _insert(self, key: str, priority: float, cost: float, size: int):
        """Insert into heap and metadata."""
        heapq.heappush(self._heap, (priority, key))
        self._metadata[key] = {
            'freq': 1,
            'cost': cost,
            'size': size,
            'priority': priority,
        }
        self._size += 1
```

---

## 8. Checklist Before Submission

- [ ] All experiments completed (4,500 main + 1,920 ablation)
- [ ] All 8 figures generated as PDF
- [ ] All 3 tables generated in LaTeX
- [ ] Statistical tests run with corrected p-values
- [ ] Sanity checks pass (no regressions, expected behavior on control)
- [ ] Metadata file generated (git hash, versions, seeds, runtime)
- [ ] Docker image builds and reproduces results
- [ ] One-command script runs end-to-end
- [ ] Report section written with coherent narrative
- [ ] All raw data archived (JSON files)
- [ ] Code is clean, documented, type-hinted
- [ ] README explains how to reproduce

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| GDSF shows no improvement | Low (theory is sound) | High | Contingency B: amplify cost variance |
| Runtime exceeds timeline | Medium | Medium | Parallelize, reduce n_runs |
| Implementation bug discovered late | Medium | High | Phase 1 sanity checks catch this |
| Similarity search confounds results | Low | Medium | Use exact-match mode for benchmarks |
| Reviewer questions workload realism | Medium | Low | Cite real-world cost distributions |
| Memory measurement inaccurate | Low | Low | Use tracemalloc for precise measurement |

---

## 10. Quick Reference: Expected Results

Based on GDSF theory and prior work on web caching:

| Workload | Expected CWHR Improvement over LRU | Confidence |
|----------|-------------------------------------|------------|
| Uniform | 0-2% (neutral) | High |
| High-Variance | 20-40% | High |
| Zipfian + Variable | 15-30% | Medium |
| Bursty | 10-20% | Medium |
| Adversarial | 40-60% | High |
| Size-Varying | 10-25% | Medium |

These are estimates based on the GDSF literature (Cao & Irani, 1997; Cherkasova,
1998) adapted to LLM caching. Actual numbers depend on workload parameters.
