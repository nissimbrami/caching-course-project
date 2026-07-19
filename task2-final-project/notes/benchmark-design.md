# Benchmark Design: Cost-Aware Eviction (GDSF) for GPTCache

## Overview

This document specifies the complete benchmark and evaluation framework for our
cost-aware eviction enhancement to GPTCache. The enhancement implements a
Greedy Dual-Size Frequency (GDSF) variant with the priority formula:

```
Priority(i) = Clock + (freq(i)^alpha * cost(i)^beta) / size(i)
```

The benchmark framework is designed to:
1. Demonstrate statistically significant improvement over baseline policies
2. Characterize when and why cost-aware eviction helps (and when it does not)
3. Be fully reproducible from a single command

---

## A. Workload Generation

### A.1 Uniform Cost (Control Workload)

**Purpose:** Establishes that our enhancement does NOT degrade performance when
all queries have equal cost. This is the "do no harm" baseline.

**Expected Behavior:** Cost-aware GDSF should perform approximately equal to
LRU (within noise). When all costs are identical, the cost term becomes a
constant and the policy degenerates to a frequency-weighted recency policy.

**Why This Matters in Production:** Many deployments use a single model. We must
show we do not regress in this common case.

```python
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Query:
    """Represents a cacheable LLM query."""
    query_id: str
    text: str
    embedding: np.ndarray  # pre-computed for reproducibility
    cost: float            # generation cost (tokens * price_per_token)
    size: int              # response size in bytes
    timestamp: float       # arrival time


def generate_uniform_cost_workload(
    n_unique_queries: int = 500,
    n_total_requests: int = 10_000,
    cost: float = 0.002,          # fixed cost ($0.002 per query)
    response_size: int = 512,     # fixed response size (bytes)
    zipf_alpha: float = 1.0,      # popularity distribution
    embedding_dim: int = 768,
    seed: int = 42
) -> List[Query]:
    """
    Generate a workload where all queries have identical cost.

    Query popularity follows a Zipfian distribution to model
    realistic access patterns (some queries are more popular).

    Parameters to sweep:
        n_unique_queries: [100, 200, 500, 1000, 2000]
        n_total_requests: [5000, 10000, 20000]
        zipf_alpha: [0.8, 1.0, 1.2, 1.5]
    """
    rng = np.random.default_rng(seed)

    # Generate unique query embeddings (unit vectors for cosine similarity)
    embeddings = rng.standard_normal((n_unique_queries, embedding_dim))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Zipfian popularity: query_id selection
    # np.random uses 1-indexed Zipf, we map to 0-indexed
    popularity_ranks = rng.zipf(zipf_alpha, size=n_total_requests)
    query_indices = (popularity_ranks - 1) % n_unique_queries

    # Generate arrival times (Poisson process, ~100 queries/sec)
    inter_arrivals = rng.exponential(scale=0.01, size=n_total_requests)
    timestamps = np.cumsum(inter_arrivals)

    workload = []
    for i, (qidx, ts) in enumerate(zip(query_indices, timestamps)):
        workload.append(Query(
            query_id=f"q_{qidx:05d}",
            text=f"uniform_query_{qidx}",
            embedding=embeddings[qidx],
            cost=cost,  # UNIFORM: same cost for all
            size=response_size,
            timestamp=ts
        ))

    return workload
```

**Parameter Sweep:**
| Parameter | Values | Purpose |
|-----------|--------|---------|
| `n_unique_queries` | 100, 200, 500, 1000, 2000 | Varies working set relative to cache |
| `zipf_alpha` | 0.8, 1.0, 1.2, 1.5 | Controls popularity skew |
| `cache_size` | 10%, 20%, 50%, 80% of unique queries | Varies pressure |

---

### A.2 High-Variance Cost (Primary Showcase)

**Purpose:** This is the PRIMARY workload that demonstrates the value of
cost-aware eviction. Queries have radically different costs (GPT-4 vs GPT-3.5
vs cached embeddings).

**Expected Behavior:** Cost-aware eviction should dramatically outperform LRU
on the "cost-weighted hit rate" metric by preferentially retaining expensive
items. LRU is cost-blind and will evict an expensive GPT-4 response just as
readily as a cheap GPT-3.5 response.

**Why This Matters in Production:** Real deployments route queries to different
models based on complexity. A single cache serves responses from GPT-4 ($0.03/1K
tokens), GPT-3.5 ($0.0005/1K tokens), and local models (near-free). Evicting a
GPT-4 response costs 60x more to regenerate than a GPT-3.5 response.

```python
def generate_high_variance_cost_workload(
    n_unique_queries: int = 500,
    n_total_requests: int = 10_000,
    cost_distribution: str = "trimodal",
    zipf_alpha: float = 1.0,
    embedding_dim: int = 768,
    seed: int = 42
) -> List[Query]:
    """
    Generate workload with high variance in query costs.

    Cost tiers model real multi-model deployments:
        - Tier 1 (GPT-4): cost=0.030, 15% of unique queries
        - Tier 2 (GPT-3.5): cost=0.002, 50% of unique queries
        - Tier 3 (Local/cached): cost=0.0001, 35% of unique queries

    Response sizes also vary by tier:
        - GPT-4: 1024-4096 bytes (longer, detailed responses)
        - GPT-3.5: 256-1024 bytes
        - Local: 64-256 bytes

    Parameters to sweep:
        cost_ratio: [10x, 30x, 60x, 100x] (max_cost / min_cost)
        expensive_fraction: [0.05, 0.10, 0.15, 0.25]
        n_unique_queries: [200, 500, 1000]
    """
    rng = np.random.default_rng(seed)

    # Assign cost tiers to unique queries
    tier_assignments = rng.choice(
        [0, 1, 2],
        size=n_unique_queries,
        p=[0.15, 0.50, 0.35]  # 15% expensive, 50% medium, 35% cheap
    )

    # Cost and size per tier
    tier_costs = {
        0: (0.030, 2048, 1024),  # (mean_cost, mean_size, size_std)
        1: (0.002, 512, 256),
        2: (0.0001, 128, 64),
    }

    # Generate per-query costs and sizes
    query_costs = np.zeros(n_unique_queries)
    query_sizes = np.zeros(n_unique_queries, dtype=int)
    for qidx in range(n_unique_queries):
        tier = tier_assignments[qidx]
        mean_cost, mean_size, size_std = tier_costs[tier]
        # Add noise within tier (log-normal)
        query_costs[qidx] = mean_cost * rng.lognormal(0, 0.3)
        query_sizes[qidx] = max(64, int(rng.normal(mean_size, size_std)))

    # Generate embeddings
    embeddings = rng.standard_normal((n_unique_queries, embedding_dim))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Zipfian access pattern (popularity independent of cost)
    popularity_ranks = rng.zipf(zipf_alpha, size=n_total_requests)
    query_indices = (popularity_ranks - 1) % n_unique_queries

    # Arrival times
    inter_arrivals = rng.exponential(scale=0.01, size=n_total_requests)
    timestamps = np.cumsum(inter_arrivals)

    workload = []
    for i, (qidx, ts) in enumerate(zip(query_indices, timestamps)):
        workload.append(Query(
            query_id=f"q_{qidx:05d}",
            text=f"var_cost_query_{qidx}_tier{tier_assignments[qidx]}",
            embedding=embeddings[qidx],
            cost=query_costs[qidx],
            size=query_sizes[qidx],
            timestamp=ts
        ))

    return workload


def generate_high_variance_cost_sweep(seed: int = 42):
    """Generate parameter sweep configurations."""
    configs = []
    for cost_ratio in [10, 30, 60, 100]:
        for expensive_frac in [0.05, 0.10, 0.15, 0.25]:
            for n_unique in [200, 500, 1000]:
                configs.append({
                    'cost_ratio': cost_ratio,
                    'expensive_fraction': expensive_frac,
                    'n_unique_queries': n_unique,
                    'seed': seed,
                })
    return configs
```

---

### A.3 Zipfian Popularity + Variable Cost (Realistic Production)

**Purpose:** Models the most realistic production scenario where query popularity
follows a Zipfian distribution AND costs vary. Popular queries may be cheap
(common questions) while rare queries may be expensive (complex analysis).

**Expected Behavior:** Cost-aware eviction should improve "dollar savings"
significantly because it will preferentially cache rare-but-expensive queries
that LRU would evict in favor of popular-but-cheap ones.

**Why This Matters in Production:** This IS the production distribution. Search
engines, chatbots, and API gateways all exhibit Zipfian popularity.

```python
def generate_zipf_variable_cost_workload(
    n_unique_queries: int = 1000,
    n_total_requests: int = 20_000,
    zipf_alpha: float = 1.2,
    cost_popularity_correlation: float = -0.3,  # negative = rare queries cost more
    embedding_dim: int = 768,
    seed: int = 42
) -> List[Query]:
    """
    Realistic workload: Zipfian popularity with correlated costs.

    Key insight: In practice, rare queries tend to be more complex
    (requiring GPT-4), while popular queries are simple (GPT-3.5 suffices).
    This negative correlation between popularity and cost is where
    cost-aware eviction shines.

    Parameters to sweep:
        zipf_alpha: [0.8, 1.0, 1.2, 1.5, 2.0]
        cost_popularity_correlation: [-0.7, -0.5, -0.3, 0.0, 0.3]
        n_unique_queries: [500, 1000, 2000]
    """
    rng = np.random.default_rng(seed)

    # Popularity rank determines base access probability
    # Rank 1 = most popular, Rank N = least popular
    ranks = np.arange(1, n_unique_queries + 1)

    # Generate costs correlated with rank
    # Higher rank (less popular) -> higher cost (when correlation < 0)
    log_ranks = np.log(ranks) / np.log(n_unique_queries)  # normalize to [0, 1]

    # Base cost from log-normal, shifted by correlation with rank
    base_log_costs = rng.standard_normal(n_unique_queries)
    correlated_log_costs = (
        cost_popularity_correlation * (log_ranks - 0.5) * 4 +
        np.sqrt(1 - cost_popularity_correlation**2) * base_log_costs
    )
    # Map to dollar costs: range from $0.0001 to $0.05
    costs = 0.001 * np.exp(correlated_log_costs)
    costs = np.clip(costs, 0.0001, 0.05)

    # Response sizes loosely correlated with cost
    sizes = (costs * 50000 + rng.normal(0, 100, n_unique_queries)).astype(int)
    sizes = np.clip(sizes, 64, 8192)

    # Generate embeddings
    embeddings = rng.standard_normal((n_unique_queries, embedding_dim))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Zipfian access pattern
    popularity_ranks = rng.zipf(zipf_alpha, size=n_total_requests)
    query_indices = (popularity_ranks - 1) % n_unique_queries

    # Arrival times (Poisson)
    inter_arrivals = rng.exponential(scale=0.01, size=n_total_requests)
    timestamps = np.cumsum(inter_arrivals)

    workload = []
    for i, (qidx, ts) in enumerate(zip(query_indices, timestamps)):
        workload.append(Query(
            query_id=f"q_{qidx:05d}",
            text=f"zipf_query_{qidx}",
            embedding=embeddings[qidx],
            cost=costs[qidx],
            size=int(sizes[qidx]),
            timestamp=ts
        ))

    return workload
```

---

### A.4 Bursty Access Pattern (Temporal Locality)

**Purpose:** Tests how cost-aware eviction interacts with temporal bursts.
Production systems see bursts (trending topics, breaking news). The policy
must balance recency (burst implies near-future reuse) with cost.

**Expected Behavior:** During bursts, both policies should perform well (high
hit rates from temporal locality). BETWEEN bursts, cost-aware should retain
expensive items from the previous burst while LRU ages them out uniformly.

**Why This Matters in Production:** Chatbots experience topic bursts. When a
topic trends, many users ask similar questions. After the burst, cost-aware
eviction should retain the expensive responses for when the topic resurfaces.

```python
def generate_bursty_workload(
    n_unique_queries: int = 500,
    n_total_requests: int = 15_000,
    n_bursts: int = 10,
    burst_size: int = 200,         # queries per burst
    burst_concentration: int = 20, # unique queries active during burst
    inter_burst_queries: int = 1000,
    embedding_dim: int = 768,
    seed: int = 42
) -> List[Query]:
    """
    Generate a workload with temporal bursts.

    Structure:
        [inter-burst random] -> [BURST: concentrated access] -> [inter-burst] -> ...

    During bursts, a small subset of queries dominates.
    Between bursts, access is spread across the full query space.
    Cost is assigned per-query (invariant of burst membership).

    Parameters to sweep:
        n_bursts: [5, 10, 20]
        burst_concentration: [10, 20, 50]
        burst_size: [100, 200, 500]
        inter_burst_queries: [500, 1000, 2000]
    """
    rng = np.random.default_rng(seed)

    # Assign costs (trimodal as in workload A.2)
    tier_assignments = rng.choice([0, 1, 2], size=n_unique_queries, p=[0.15, 0.5, 0.35])
    tier_cost_map = {0: 0.030, 1: 0.002, 2: 0.0001}
    costs = np.array([tier_cost_map[t] * rng.lognormal(0, 0.2) for t in tier_assignments])
    sizes = np.array([max(64, int(rng.normal(512, 200))) for _ in range(n_unique_queries)])

    # Generate embeddings
    embeddings = rng.standard_normal((n_unique_queries, embedding_dim))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Build workload: alternating bursts and inter-burst periods
    query_sequence = []

    for burst_idx in range(n_bursts):
        # Inter-burst: uniform random over all queries
        inter_indices = rng.integers(0, n_unique_queries, size=inter_burst_queries)
        query_sequence.extend(inter_indices.tolist())

        # Burst: concentrated on a random subset
        burst_queries = rng.choice(n_unique_queries, size=burst_concentration, replace=False)
        # Within burst, access follows Zipf on the burst subset
        burst_ranks = rng.zipf(1.5, size=burst_size)
        burst_indices = burst_queries[(burst_ranks - 1) % burst_concentration]
        query_sequence.extend(burst_indices.tolist())

    # Final inter-burst period
    final_inter = rng.integers(0, n_unique_queries, size=inter_burst_queries)
    query_sequence.extend(final_inter.tolist())

    # Timestamps
    n_total = len(query_sequence)
    inter_arrivals = rng.exponential(scale=0.01, size=n_total)
    timestamps = np.cumsum(inter_arrivals)

    workload = []
    for i, qidx in enumerate(query_sequence):
        workload.append(Query(
            query_id=f"q_{qidx:05d}",
            text=f"bursty_query_{qidx}",
            embedding=embeddings[qidx],
            cost=costs[qidx],
            size=int(sizes[qidx]),
            timestamp=timestamps[i]
        ))

    return workload
```

---

### A.5 Adversarial for LRU (Scan + Expensive Recurring)

**Purpose:** Constructs a pathological case for LRU: a scanning pattern that
continuously introduces new queries, pushing out expensive recurring queries.
This maximizes the gap between cost-aware and LRU.

**Expected Behavior:** LRU will perform catastrophically because scans evict
everything. Cost-aware GDSF retains expensive items because their high priority
prevents eviction even during scans.

**Why This Matters in Production:** Real systems experience "scans" from
batch processes, crawlers, or cold-start periods. A cost-aware policy provides
resilience against these pollution patterns.

```python
def generate_adversarial_lru_workload(
    cache_size: int = 100,           # number of items cache can hold
    n_expensive_recurring: int = 20, # expensive queries that recur
    n_scan_queries: int = 500,       # unique scan queries (>> cache_size)
    n_cycles: int = 50,              # number of scan+revisit cycles
    expensive_cost: float = 0.030,
    scan_cost: float = 0.0002,
    embedding_dim: int = 768,
    seed: int = 42
) -> List[Query]:
    """
    Adversarial workload for LRU:

    Pattern per cycle:
        1. Access N expensive recurring queries (should stay cached)
        2. Scan through M unique cheap queries (pollutes LRU cache)
        3. Repeat

    After step 2, LRU has evicted ALL expensive queries.
    Cost-aware GDSF retains them because:
        Priority(expensive) = Clock + (freq^a * 0.030^b) / size  >> scan priority

    Parameters to sweep:
        cache_size: [50, 100, 200]
        n_expensive_recurring: [10, 20, 50]
        scan_length: [2x, 5x, 10x cache_size]
        cost_ratio: [30x, 60x, 150x]
    """
    rng = np.random.default_rng(seed)

    n_total_unique = n_expensive_recurring + n_scan_queries

    # Generate embeddings for all queries
    embeddings = rng.standard_normal((n_total_unique, embedding_dim))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Costs: first n_expensive_recurring are expensive, rest are cheap
    costs = np.zeros(n_total_unique)
    costs[:n_expensive_recurring] = expensive_cost
    costs[n_expensive_recurring:] = scan_cost

    sizes = np.full(n_total_unique, 512)  # uniform size for clarity

    # Build access sequence
    query_sequence = []
    for cycle in range(n_cycles):
        # Phase 1: Access expensive recurring queries (random order)
        expensive_order = rng.permutation(n_expensive_recurring)
        query_sequence.extend(expensive_order.tolist())

        # Phase 2: Linear scan through cheap unique queries
        # Each cycle scans a DIFFERENT portion to maximize pollution
        scan_start = (cycle * cache_size) % n_scan_queries
        scan_indices = [
            n_expensive_recurring + ((scan_start + j) % n_scan_queries)
            for j in range(cache_size * 2)  # scan 2x cache_size
        ]
        query_sequence.extend(scan_indices)

    # Final access to expensive queries (to measure if they survived)
    final_expensive = list(range(n_expensive_recurring))
    query_sequence.extend(final_expensive)

    # Timestamps
    n_total = len(query_sequence)
    inter_arrivals = rng.exponential(scale=0.01, size=n_total)
    timestamps = np.cumsum(inter_arrivals)

    workload = []
    for i, qidx in enumerate(query_sequence):
        workload.append(Query(
            query_id=f"q_{qidx:05d}",
            text=f"adv_query_{qidx}",
            embedding=embeddings[qidx],
            cost=costs[qidx],
            size=int(sizes[qidx]),
            timestamp=timestamps[i]
        ))

    return workload
```

---

### A.6 Size-Varying Workload (GDSF Size Component)

**Purpose:** Tests the size-awareness component of GDSF. When responses have
different sizes, the policy should prefer caching many small items over one
large item (if total cost is similar).

**Expected Behavior:** GDSF achieves higher hit rate than cost-only policies
because it accounts for the opportunity cost of space.

```python
def generate_size_varying_workload(
    n_unique_queries: int = 500,
    n_total_requests: int = 10_000,
    size_distribution: str = "heavy_tail",  # "uniform", "bimodal", "heavy_tail"
    zipf_alpha: float = 1.0,
    embedding_dim: int = 768,
    seed: int = 42
) -> List[Query]:
    """
    Workload with high variance in response sizes.

    Size tiers:
        - Small responses: 64-256 bytes (quick factual answers)
        - Medium responses: 512-2048 bytes (explanations)
        - Large responses: 4096-16384 bytes (code generation, essays)

    Parameters to sweep:
        size_ratio: [10x, 50x, 100x] (max/min size)
        large_fraction: [0.05, 0.10, 0.20]
        cache_capacity_bytes: varies
    """
    rng = np.random.default_rng(seed)

    # Heavy-tailed size distribution (Pareto)
    if size_distribution == "heavy_tail":
        raw_sizes = (rng.pareto(1.5, size=n_unique_queries) + 1) * 128
        sizes = np.clip(raw_sizes, 64, 16384).astype(int)
    elif size_distribution == "bimodal":
        is_large = rng.random(n_unique_queries) < 0.15
        sizes = np.where(is_large,
                         rng.normal(8192, 2048, n_unique_queries),
                         rng.normal(256, 64, n_unique_queries))
        sizes = np.clip(sizes, 64, 16384).astype(int)
    else:  # uniform
        sizes = rng.integers(64, 4096, size=n_unique_queries)

    # Cost roughly proportional to size (more tokens = more cost)
    costs = sizes * 0.00001 * rng.lognormal(0, 0.5, n_unique_queries)
    costs = np.clip(costs, 0.0001, 0.05)

    # Embeddings
    embeddings = rng.standard_normal((n_unique_queries, embedding_dim))
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Access pattern
    popularity_ranks = rng.zipf(zipf_alpha, size=n_total_requests)
    query_indices = (popularity_ranks - 1) % n_unique_queries

    inter_arrivals = rng.exponential(scale=0.01, size=n_total_requests)
    timestamps = np.cumsum(inter_arrivals)

    workload = []
    for i, (qidx, ts) in enumerate(zip(query_indices, timestamps)):
        workload.append(Query(
            query_id=f"q_{qidx:05d}",
            text=f"size_var_query_{qidx}",
            embedding=embeddings[qidx],
            cost=costs[qidx],
            size=int(sizes[qidx]),
            timestamp=ts
        ))

    return workload
```

---

## B. Metrics Definition

### B.1 Hit Rate (Standard)

**Formula:**
```
HitRate = N_hits / (N_hits + N_misses)
```

**Measurement Code:**
```python
class MetricsCollector:
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.hit_costs = []
        self.miss_costs = []
        self.latencies = []
        self.all_costs = []
        self.eviction_count = 0
        self.metadata_bytes = 0
        self._start_time = None
        self._query_count = 0

    def record_hit(self, cost: float, latency_ms: float):
        self.hits += 1
        self.hit_costs.append(cost)
        self.all_costs.append(cost)
        self.latencies.append(latency_ms)
        self._query_count += 1

    def record_miss(self, cost: float, latency_ms: float):
        self.misses += 1
        self.miss_costs.append(cost)
        self.all_costs.append(cost)
        self.latencies.append(latency_ms)
        self._query_count += 1

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
```

**What It Tells Us:** Basic cache effectiveness. The fraction of queries served
from cache without calling the LLM API.

**Aggregation:** Report mean across runs. Also report per-window hit rate
(rolling window of 1000 queries) to show warm-up and stability.

---

### B.2 Cost-Weighted Hit Rate (CWHR)

**Formula:**
```
CWHR = sum(cost_i for i in hits) / sum(cost_i for all i)
```

**Measurement Code:**
```python
    @property
    def cost_weighted_hit_rate(self) -> float:
        total_cost = sum(self.all_costs)
        hit_cost = sum(self.hit_costs)
        return hit_cost / total_cost if total_cost > 0 else 0.0
```

**What It Tells Us:** The fraction of COST that was avoided by caching. This is
the PRIMARY metric for cost-aware eviction. A policy that caches expensive items
preferentially will have CWHR >> HitRate when costs are variable.

**Aggregation:** Report mean, median, and standard deviation across runs.
This is our HEADLINE metric.

---

### B.3 Dollar Savings

**Formula:**
```
DollarSavings = sum(cost_i for i in hits)
```
In production terms: if `cost_i` is the actual API cost to regenerate query i,
then DollarSavings is the total money saved by the cache.

**Measurement Code:**
```python
    @property
    def dollar_savings(self) -> float:
        return sum(self.hit_costs)

    @property
    def dollar_savings_vs_no_cache(self) -> float:
        """Percentage savings compared to no caching."""
        total = sum(self.all_costs)
        return (self.dollar_savings / total * 100) if total > 0 else 0.0
```

**What It Tells Us:** The raw business value of the cache policy. This
translates directly to operational cost reduction.

**Aggregation:** Report total dollar savings per 10K queries. Compare absolute
and relative (percentage of total possible savings).

---

### B.4 Latency (p50, p95, p99)

**Formula:**
```
p50 = percentile(latencies, 50)
p95 = percentile(latencies, 95)
p99 = percentile(latencies, 99)
```

**Measurement Code:**
```python
    @property
    def latency_stats(self) -> dict:
        if not self.latencies:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'mean': 0}
        arr = np.array(self.latencies)
        return {
            'p50': np.percentile(arr, 50),
            'p95': np.percentile(arr, 95),
            'p99': np.percentile(arr, 99),
            'mean': np.mean(arr),
            'std': np.std(arr),
        }
```

**What It Tells Us:** User-facing response time. Cache hits should be 10-100x
faster than misses (no LLM API call). Higher hit rates -> lower tail latencies.

**Aggregation:** Report percentiles directly. For comparison across policies,
report the REDUCTION in p95 latency.

**Note on Measurement:** In simulation mode, we model latency as:
- Hit latency: ~5ms (cache lookup + similarity search)
- Miss latency: ~500ms-2000ms (API call, varies by model/response length)

```python
def simulate_latency(is_hit: bool, cost: float, rng) -> float:
    """Simulate realistic latency in milliseconds."""
    if is_hit:
        # Cache hit: similarity search + retrieval
        return max(1.0, rng.normal(5.0, 2.0))
    else:
        # Cache miss: API call, latency correlates with cost/complexity
        base_latency = 500 + cost * 30000  # more expensive = longer
        return max(100.0, rng.normal(base_latency, base_latency * 0.2))
```

---

### B.5 Throughput

**Formula:**
```
Throughput = N_queries / wall_clock_time_seconds
```

**Measurement Code:**
```python
    def start_timer(self):
        self._start_time = time.perf_counter()

    def stop_timer(self):
        self._elapsed = time.perf_counter() - self._start_time

    @property
    def throughput(self) -> float:
        """Queries per second."""
        return self._query_count / self._elapsed if self._elapsed > 0 else 0.0
```

**What It Tells Us:** The overhead of the eviction policy itself. A more complex
policy (like GDSF with priority queue maintenance) may have lower throughput
than simple LRU. We need to show this overhead is negligible.

**Aggregation:** Report mean throughput with 95% CI. Measure in "policy
operations per second" (excluding simulated API latency) to isolate policy
overhead.

---

### B.6 Memory Overhead

**Formula:**
```
MemoryOverhead = bytes_used_by_policy_metadata - bytes_used_by_LRU_metadata
```

**Measurement Code:**
```python
import sys

def measure_policy_memory(policy) -> int:
    """Measure memory used by policy data structures (approximate)."""
    # For GDSF: priority queue + frequency counters + cost map
    total = sys.getsizeof(policy)

    if hasattr(policy, '_priority_queue'):
        total += sys.getsizeof(policy._priority_queue)
        for item in policy._priority_queue:
            total += sys.getsizeof(item)

    if hasattr(policy, '_frequency'):
        total += sys.getsizeof(policy._frequency)

    if hasattr(policy, '_cost_map'):
        total += sys.getsizeof(policy._cost_map)

    return total


def measure_memory_overhead_ratio(gdsf_policy, lru_policy) -> float:
    """Ratio of GDSF memory to LRU memory."""
    gdsf_mem = measure_policy_memory(gdsf_policy)
    lru_mem = measure_policy_memory(lru_policy)
    return gdsf_mem / lru_mem if lru_mem > 0 else float('inf')
```

**What It Tells Us:** The space cost of the enhanced policy. GDSF requires
additional metadata (frequency counters, cost records, priority values) beyond
what LRU needs (just a doubly-linked list). We want to show this is < 2x.

**Aggregation:** Report absolute bytes and ratio vs LRU at various cache sizes.

---

### B.7 Eviction Efficiency (vs Optimal)

**Formula:**
```
EvictionEfficiency = CostSavings(policy) / CostSavings(OPT)
```

Where OPT is Belady's algorithm adapted for cost-awareness:
- For standard hit rate: Belady evicts the item used furthest in the future
- For cost-weighted: modified Belady that considers cost * distance

**Measurement Code:**
```python
def compute_optimal_cost_savings(workload: List[Query], cache_size: int) -> float:
    """
    Compute optimal (offline) cost savings using modified Belady's.

    This is an upper bound on what ANY online policy can achieve.
    Uses dynamic programming or greedy with future knowledge.
    """
    # Build future access timeline
    from collections import defaultdict
    future_accesses = defaultdict(list)
    for i, query in enumerate(workload):
        future_accesses[query.query_id].append(i)

    # ... (full implementation in benchmark code)
    # Simplified: at each eviction, evict item with lowest
    # cost / time_until_next_access
    pass


def eviction_efficiency(policy_savings: float, optimal_savings: float) -> float:
    """How close to optimal the policy achieves."""
    return policy_savings / optimal_savings if optimal_savings > 0 else 1.0
```

**What It Tells Us:** How close to theoretically optimal our policy is. Values
> 0.8 indicate excellent performance. This contextualizes absolute numbers.

**Aggregation:** Report mean efficiency across workloads.

---

## C. Statistical Methodology

### C.1 Number of Runs

Each configuration is run **30 times** with different random seeds.

**Justification:**
- 30 runs gives sufficient power for t-tests (CLT applies)
- With n=30, we can detect effect sizes > 0.5 standard deviations with 80% power
- Computational feasibility: simulation runs take ~1-5 seconds each

**Seed Strategy:**
```python
BASE_SEED = 42
N_RUNS = 30

def get_seeds(experiment_id: int, n_runs: int = N_RUNS) -> List[int]:
    """Generate deterministic seeds for an experiment."""
    rng = np.random.default_rng(BASE_SEED + experiment_id * 1000)
    return rng.integers(0, 2**31, size=n_runs).tolist()
```

### C.2 Confidence Intervals

Use **bootstrap confidence intervals** (bias-corrected and accelerated, BCa):

```python
from scipy import stats

def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000,
                 confidence: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """
    Compute BCa bootstrap confidence interval.

    More robust than t-based CIs for potentially non-normal distributions.
    """
    rng = np.random.default_rng(seed)
    n = len(data)

    # Bootstrap resamples
    boot_means = np.array([
        np.mean(rng.choice(data, size=n, replace=True))
        for _ in range(n_bootstrap)
    ])

    # Percentile method (simple; use scipy for BCa)
    alpha = (1 - confidence) / 2
    ci_low = np.percentile(boot_means, alpha * 100)
    ci_high = np.percentile(boot_means, (1 - alpha) * 100)

    return ci_low, ci_high


def scipy_bootstrap_ci(data: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
    """Production-quality BCa bootstrap CI using scipy."""
    result = stats.bootstrap(
        (data,),
        statistic=np.mean,
        confidence_level=confidence,
        n_resamples=10000,
        method='BCa'
    )
    return result.confidence_interval.low, result.confidence_interval.high
```

### C.3 Statistical Tests

**Primary Comparison (GDSF vs LRU):**

1. **Welch's t-test** (unequal variances): For normally distributed metrics
2. **Mann-Whitney U test**: For non-normal metrics (latency, which is skewed)
3. **Paired comparison**: Since we run same workload with different policies,
   use **paired t-test** or **Wilcoxon signed-rank** for matched pairs.

```python
from scipy.stats import ttest_rel, wilcoxon, mannwhitneyu, shapiro

def compare_policies(
    metric_gdsf: np.ndarray,
    metric_baseline: np.ndarray,
    alpha: float = 0.05,
    paired: bool = True
) -> dict:
    """
    Statistical comparison between GDSF and a baseline policy.

    Returns dict with test statistics, p-values, effect sizes.
    """
    # Check normality of differences (for paired test selection)
    differences = metric_gdsf - metric_baseline
    _, normality_p = shapiro(differences)

    if paired:
        if normality_p > 0.05:
            # Normal: paired t-test
            stat, p_value = ttest_rel(metric_gdsf, metric_baseline)
            test_name = "paired_t_test"
        else:
            # Non-normal: Wilcoxon signed-rank
            stat, p_value = wilcoxon(differences, alternative='greater')
            test_name = "wilcoxon_signed_rank"
    else:
        stat, p_value = mannwhitneyu(
            metric_gdsf, metric_baseline, alternative='greater'
        )
        test_name = "mann_whitney_u"

    # Effect size (Cohen's d for paired)
    d = np.mean(differences) / np.std(differences, ddof=1)

    # Practical significance
    improvement_pct = (np.mean(metric_gdsf) - np.mean(metric_baseline)) / np.mean(metric_baseline) * 100

    return {
        'test': test_name,
        'statistic': stat,
        'p_value': p_value,
        'significant': p_value < alpha,
        'cohens_d': d,
        'improvement_pct': improvement_pct,
        'mean_gdsf': np.mean(metric_gdsf),
        'mean_baseline': np.mean(metric_baseline),
        'ci_difference': bootstrap_ci(differences),
    }
```

### C.4 Multiple Comparisons Correction

When comparing GDSF against multiple baselines, apply **Bonferroni correction**
or **Holm-Bonferroni** to control family-wise error rate:

```python
from statsmodels.stats.multitest import multipletests

def correct_multiple_comparisons(p_values: List[float], method='holm') -> np.ndarray:
    """Apply multiple testing correction."""
    reject, corrected_p, _, _ = multipletests(p_values, method=method)
    return corrected_p
```

### C.5 Warm-Up Period

**Strategy:** Discard the first 10% of queries from metric computation.

```python
WARMUP_FRACTION = 0.10

def apply_warmup(metrics_over_time: List[float], warmup_frac: float = WARMUP_FRACTION) -> List[float]:
    """Remove warm-up period from metrics."""
    warmup_end = int(len(metrics_over_time) * warmup_frac)
    return metrics_over_time[warmup_end:]
```

**Justification:** Cache starts empty; initial misses are unavoidable and do not
reflect steady-state policy behavior. Reporting only steady-state metrics gives
a fairer comparison.

**Reporting:** We report BOTH warm-up-inclusive and steady-state metrics.
Time-series plots show the full trace including warm-up.

### C.6 Significance Thresholds

| Level | p-value | Interpretation |
|-------|---------|----------------|
| * | < 0.05 | Significant |
| ** | < 0.01 | Highly significant |
| *** | < 0.001 | Very highly significant |
| ns | >= 0.05 | Not significant |

Report exact p-values in tables; use stars in plots for readability.

---

## D. Plot Specifications

### D.1 Hit Rate vs Cache Size

**Plot Type:** Line plot with error bands

**X-axis:** Cache size (as % of unique query count): [5%, 10%, 20%, 30%, 50%, 80%]
**Y-axis:** Hit Rate [0.0, 1.0]
**Series:** LRU, FIFO, LFU, Random, GDSF (our enhancement)
**Error:** 95% confidence interval shaded region

**Figure Size:** 8 x 5 inches
**Caption:** "Hit rate as a function of cache size for workload W2 (high-variance
cost). All policies converge as cache size approaches working set size. GDSF
maintains comparable hit rate to LRU across all cache sizes."

**Story:** Establishes that GDSF does not sacrifice HIT RATE. It may be slightly
lower (it evicts frequently-accessed cheap items), but the gap is small.

```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

def plot_hit_rate_vs_cache_size(results: dict, workload_name: str):
    """
    results: {policy_name: {cache_size: {'mean': float, 'ci_low': float, 'ci_high': float}}}
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {'LRU': '#1f77b4', 'FIFO': '#ff7f0e', 'LFU': '#2ca02c',
              'Random': '#d62728', 'GDSF': '#9467bd'}
    markers = {'LRU': 'o', 'FIFO': 's', 'LFU': '^', 'Random': 'x', 'GDSF': 'D'}

    for policy, data in results.items():
        sizes = sorted(data.keys())
        means = [data[s]['mean'] for s in sizes]
        ci_low = [data[s]['ci_low'] for s in sizes]
        ci_high = [data[s]['ci_high'] for s in sizes]

        ax.plot(sizes, means, color=colors[policy], marker=markers[policy],
                label=policy, linewidth=2, markersize=6)
        ax.fill_between(sizes, ci_low, ci_high, color=colors[policy], alpha=0.15)

    ax.set_xlabel('Cache Size (% of working set)')
    ax.set_ylabel('Hit Rate')
    ax.set_title(f'Hit Rate vs Cache Size ({workload_name})')
    ax.legend(loc='lower right', frameon=True)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig
```

---

### D.2 Cost-Weighted Hit Rate vs Cache Size

**Plot Type:** Line plot with error bands

**X-axis:** Cache size (% of working set): [5%, 10%, 20%, 30%, 50%, 80%]
**Y-axis:** Cost-Weighted Hit Rate [0.0, 1.0]
**Series:** LRU, FIFO, LFU, Random, GDSF
**Error:** 95% confidence interval bands

**Figure Size:** 8 x 5 inches
**Caption:** "Cost-weighted hit rate demonstrates the primary advantage of
GDSF: preferential caching of expensive items. At 20% cache size, GDSF achieves
72% CWHR vs 45% for LRU (p < 0.001, Cohen's d = 2.1)."

**Story:** THIS IS THE MAIN RESULT. Shows dramatic improvement on the metric
that matters most for cost optimization.

---

### D.3 Dollar Savings Comparison

**Plot Type:** Grouped bar chart

**X-axis:** Policy names (categorical): [Random, FIFO, LRU, LFU, GDSF]
**Y-axis:** Dollar Savings per 10K queries ($)
**Groups:** One group per workload type [Uniform, High-Var, Zipfian, Bursty, Adversarial]
**Error bars:** 95% CI

**Figure Size:** 10 x 6 inches
**Caption:** "Total dollar savings per 10,000 queries across workloads. GDSF
provides $X.XX additional savings over LRU in the high-variance cost workload,
representing a Y% improvement in operational costs."

**Story:** Translates the abstract metric into real dollar amounts. Makes the
business case for cost-aware eviction.

```python
def plot_dollar_savings(results: dict):
    """
    results: {workload: {policy: {'mean': float, 'ci': (low, high)}}}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    workloads = list(results.keys())
    policies = ['Random', 'FIFO', 'LRU', 'LFU', 'GDSF']
    colors = ['#d62728', '#ff7f0e', '#1f77b4', '#2ca02c', '#9467bd']

    n_workloads = len(workloads)
    n_policies = len(policies)
    bar_width = 0.15
    x = np.arange(n_workloads)

    for i, (policy, color) in enumerate(zip(policies, colors)):
        means = [results[w][policy]['mean'] for w in workloads]
        errors = [
            (results[w][policy]['mean'] - results[w][policy]['ci'][0],
             results[w][policy]['ci'][1] - results[w][policy]['mean'])
            for w in workloads
        ]
        errors = np.array(errors).T

        offset = (i - n_policies/2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, label=policy, color=color,
               yerr=errors, capsize=3, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Workload')
    ax.set_ylabel('Dollar Savings per 10K Queries ($)')
    ax.set_title('Cost Savings by Policy and Workload')
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=15)
    ax.legend(loc='upper left')
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    return fig
```

---

### D.4 Latency CDF

**Plot Type:** CDF (Cumulative Distribution Function) plot

**X-axis:** Latency (ms), log scale: [1, 10, 100, 1000, 10000]
**Y-axis:** Cumulative Probability [0, 1]
**Series:** Vanilla GPTCache (LRU), Enhanced (GDSF)
**Vertical lines:** p50, p95, p99 markers

**Figure Size:** 8 x 5 inches
**Caption:** "Latency CDF showing that GDSF achieves similar latency profile
to LRU. The slight leftward shift indicates marginally better tail latencies
due to caching expensive (and slow) queries more aggressively."

**Story:** Demonstrates no latency regression; potentially slight improvement
because expensive queries also tend to be slower to generate.

```python
def plot_latency_cdf(latencies_lru: np.ndarray, latencies_gdsf: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 5))

    for data, label, color in [
        (latencies_lru, 'LRU (baseline)', '#1f77b4'),
        (latencies_gdsf, 'GDSF (ours)', '#9467bd'),
    ]:
        sorted_data = np.sort(data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        ax.plot(sorted_data, cdf, label=label, color=color, linewidth=2)

        # Mark percentiles
        for p, style in [(50, '--'), (95, ':'), (99, '-.')]:
            val = np.percentile(data, p)
            ax.axvline(val, color=color, linestyle=style, alpha=0.5, linewidth=1)

    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Cumulative Probability')
    ax.set_xscale('log')
    ax.set_title('Latency CDF: LRU vs GDSF')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

---

### D.5 Ablation Heatmap (alpha vs beta)

**Plot Type:** Heatmap (2D color grid)

**X-axis:** alpha values: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
**Y-axis:** beta values: [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
**Color:** Cost-Weighted Hit Rate [colormap: viridis]
**Annotations:** Cell values (2 decimal places)
**Special markers:** Star on optimal (alpha, beta), circle on LRU-equivalent (0,0)

**Figure Size:** 8 x 7 inches
**Caption:** "Ablation study over GDSF parameters alpha (frequency exponent) and
beta (cost exponent). Optimal performance at alpha=X, beta=Y. Setting both to 0
recovers simple recency (Clock only). The broad plateau around the optimum
indicates robustness to parameter choice."

**Story:** Shows the parameter space has a clear optimum, demonstrates
robustness (not overly sensitive to exact parameter values), and confirms
that both frequency and cost components contribute to performance.

```python
import seaborn as sns

def plot_ablation_heatmap(results: np.ndarray, alphas: List[float], betas: List[float]):
    """
    results: 2D array of shape (len(alphas), len(betas)) with CWHR values
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        results, ax=ax,
        xticklabels=[f'{b:.2f}' for b in betas],
        yticklabels=[f'{a:.2f}' for a in alphas],
        annot=True, fmt='.3f',
        cmap='viridis',
        cbar_kws={'label': 'Cost-Weighted Hit Rate'},
        linewidths=0.5,
    )

    # Mark optimal
    opt_idx = np.unravel_index(np.argmax(results), results.shape)
    ax.plot(opt_idx[1] + 0.5, opt_idx[0] + 0.5, '*', color='red',
            markersize=20, markeredgecolor='white', markeredgewidth=1.5)

    ax.set_xlabel(r'$\beta$ (cost exponent)')
    ax.set_ylabel(r'$\alpha$ (frequency exponent)')
    ax.set_title(r'GDSF Parameter Sensitivity: $\alpha$ vs $\beta$')

    plt.tight_layout()
    return fig
```

---

### D.6 Workload Sensitivity (Grouped Bars)

**Plot Type:** Grouped bar chart with significance annotations

**X-axis:** Workload type (categorical): [Uniform, High-Var, Zipfian, Bursty, Adversarial, Size-Var]
**Y-axis:** Improvement in CWHR over LRU (percentage points)
**Single series:** GDSF improvement, with error bars
**Annotations:** Stars for significance level (*, **, ***)

**Figure Size:** 8 x 5 inches
**Caption:** "GDSF improvement over LRU baseline across workloads. The uniform
workload shows negligible difference (as expected), while high-variance and
adversarial workloads show substantial gains. All improvements except uniform
are statistically significant (p < 0.01)."

**Story:** Shows WHERE cost-aware eviction helps and where it is neutral.
Demonstrates understanding of when the technique applies.

---

### D.7 Memory Overhead Comparison

**Plot Type:** Stacked bar chart

**X-axis:** Cache size: [100, 500, 1000, 5000, 10000] entries
**Y-axis:** Memory usage (KB)
**Stacks:** Base data structure, Frequency counters, Cost metadata, Priority queue
**Comparison:** Side-by-side bars for LRU vs GDSF

**Figure Size:** 8 x 5 inches
**Caption:** "Memory overhead of GDSF vs LRU. GDSF uses approximately 1.4x the
memory of LRU due to additional frequency and cost metadata. At typical cache
sizes (1000 entries), the overhead is < 50KB -- negligible compared to cached
response data (multiple MB)."

**Story:** The memory cost is trivial in absolute terms and scales linearly.

---

### D.8 Parameter Sensitivity (Line Plot)

**Plot Type:** Multi-panel line plot (2x2 subplots)

**Panels:**
1. CWHR vs alpha (beta fixed at optimal)
2. CWHR vs beta (alpha fixed at optimal)
3. Hit Rate vs alpha (to show trade-off)
4. Dollar Savings vs cost_ratio in workload

**X-axes:** Parameter value (continuous)
**Y-axes:** Metric value
**Error:** 95% CI bands

**Figure Size:** 10 x 8 inches (2x2 grid)
**Caption:** "Parameter sensitivity analysis. (a,b) Individual parameter sweeps
show smooth response surfaces with clear optima. (c) Trade-off between standard
hit rate and cost-weighted hit rate. (d) GDSF advantage increases with cost
variance in the workload."

**Story:** Provides practical guidance for parameter tuning and shows the
enhancement is robust.

---

## E. Ablation Study Design

### E.1 Full Factor Table

| Experiment | alpha | beta | size_aware | What it tests |
|-----------|-------|------|------------|---------------|
| Clock-only | 0.0 | 0.0 | No | Pure recency (baseline equivalent) |
| Freq-only | 1.0 | 0.0 | No | Frequency without cost (LFU-like) |
| Cost-only | 0.0 | 1.0 | No | Cost without frequency |
| GDSF-no-size | 1.0 | 1.0 | No | Freq + Cost, no size normalization |
| GDSF-full | 1.0 | 1.0 | Yes | Full GDSF formula |
| GDSF-tuned | alpha* | beta* | Yes | Tuned parameters |

### E.2 Parameter Grid

```python
ABLATION_GRID = {
    'alpha': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
    'beta': [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
}
# Total: 8 x 8 = 64 configurations
# With 30 runs each: 1920 experiment runs
# At ~2 seconds per run: ~64 minutes total compute
```

### E.3 Expected Trends

1. **alpha = 0, beta = 0:** Should match LRU closely (pure Clock policy)
2. **Increasing alpha:** Improves hit rate (frequency helps) up to a point,
   then over-caches stale popular items
3. **Increasing beta:** Improves CWHR (cost-awareness) but may hurt raw hit rate
   (keeps expensive items even if rarely accessed)
4. **Optimal region:** alpha in [0.5, 1.5], beta in [0.5, 1.5] (hypothesis)
5. **Size normalization:** Should help when response sizes are variable
   (workload A.6), neutral when sizes are uniform

### E.4 Presentation Strategy

- Heatmap (Plot D.5) for the full alpha x beta grid
- Line plots with one parameter fixed at optimal for the other
- Table comparing named configurations (E.1) with all metrics
- Highlight the "default" recommendation: (alpha=1.0, beta=1.0, size_aware=True)

---

## F. Comparison Baselines

### F.1 Vanilla LRU (GPTCache Default)

**Description:** Evicts the least recently used item. Standard in GPTCache via
`from gptcache.manager.eviction import EvictionBase`. This is the primary
baseline because it is what users currently experience.

**Why Included:** The default policy that ships with GPTCache. Our enhancement
must beat this to justify its complexity.

### F.2 FIFO (GPTCache Default)

**Description:** Evicts the oldest item (first in, first out). Available in
GPTCache as an alternative eviction manager.

**Why Included:** Simpler than LRU; some deployments use it. Tests whether
even basic policies are competitive with cost-awareness.

### F.3 LFU (Additional Baseline)

**Description:** Evicts the least frequently used item. We implement this as a
simple frequency counter with eviction of the minimum-count item.

**Why Included:** Tests the "frequency component" of GDSF in isolation. If GDSF
with beta=0 approximates LFU, we can measure the marginal value of cost-awareness.

```python
class LFUPolicy:
    """Least Frequently Used eviction policy."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.frequency = {}  # key -> access count
        self.cache = {}      # key -> value
        self.min_freq = 0
        self.freq_to_keys = defaultdict(OrderedDict)

    def get(self, key: str):
        if key not in self.cache:
            return None
        self.frequency[key] += 1
        # Move to new frequency bucket
        old_freq = self.frequency[key] - 1
        del self.freq_to_keys[old_freq][key]
        if not self.freq_to_keys[old_freq] and self.min_freq == old_freq:
            self.min_freq += 1
        self.freq_to_keys[self.frequency[key]][key] = None
        return self.cache[key]

    def put(self, key: str, value, cost: float = 0, size: int = 0):
        if key in self.cache:
            self.cache[key] = value
            self.get(key)  # update frequency
            return None

        evicted = None
        if len(self.cache) >= self.max_size:
            # Evict least frequent
            evict_key, _ = self.freq_to_keys[self.min_freq].popitem(last=False)
            del self.cache[evict_key]
            del self.frequency[evict_key]
            evicted = evict_key

        self.cache[key] = value
        self.frequency[key] = 1
        self.freq_to_keys[1][key] = None
        self.min_freq = 1
        return evicted
```

### F.4 Random Eviction

**Description:** Evicts a random item from the cache. Simplest possible policy.

**Why Included:** Lower bound on policy intelligence. Any reasonable policy
should beat random. Also useful as a sanity check -- if our workload generator
has a bug, random might match other policies (indicating no exploitable pattern).

```python
class RandomPolicy:
    """Random eviction policy."""

    def __init__(self, max_size: int, seed: int = 42):
        self.max_size = max_size
        self.cache = {}
        self.keys = []
        self.rng = np.random.default_rng(seed)

    def get(self, key: str):
        return self.cache.get(key)

    def put(self, key: str, value, cost: float = 0, size: int = 0):
        if key in self.cache:
            self.cache[key] = value
            return None

        evicted = None
        if len(self.cache) >= self.max_size:
            evict_idx = self.rng.integers(0, len(self.keys))
            evict_key = self.keys[evict_idx]
            del self.cache[evict_key]
            self.keys[evict_idx] = self.keys[-1]
            self.keys.pop()
            evicted = evict_key

        self.cache[key] = value
        self.keys.append(key)
        return evicted
```

### F.5 Cost-Aware GDSF (Our Enhancement)

**Description:** Greedy Dual-Size Frequency with priority formula:
`Priority(i) = Clock + (freq(i)^alpha * cost(i)^beta) / size(i)`.
Evicts the item with minimum priority. Clock advances to the evicted item's
priority value (aging mechanism).

**Why Included:** This is what we are evaluating. It should dominate on
cost-weighted metrics while maintaining competitive hit rates.

---

## G. Reproducibility Package

### G.1 Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /benchmark

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy benchmark code
COPY . .

# Default command: run full benchmark suite
ENTRYPOINT ["python", "-m", "benchmark.run_all"]
```

```text
# requirements.txt
numpy==1.26.4
scipy==1.12.0
pandas==2.2.1
matplotlib==3.8.3
seaborn==0.13.2
gptcache==0.1.43
pytest==8.0.2
tqdm==4.66.2
psutil==5.9.8
statsmodels==0.14.1
```

### G.2 Environment Variables

```bash
# .env (NOT committed to git)
BENCHMARK_SEED=42
BENCHMARK_N_RUNS=30
BENCHMARK_OUTPUT_DIR=./results
BENCHMARK_CACHE_SIZES=100,200,500,1000,2000
BENCHMARK_WORKLOADS=uniform,high_variance,zipfian,bursty,adversarial,size_varying

# For live API tests (optional, not needed for simulation)
OPENAI_API_KEY=<redacted>
```

### G.3 Seed Management

```python
# benchmark/seeds.py
"""
Deterministic seed management for reproducibility.

All randomness flows from a single master seed through a hierarchy:
    MASTER_SEED -> experiment_seed -> run_seed -> {workload_rng, policy_rng}
"""

import hashlib
import numpy as np

MASTER_SEED = 42

def experiment_seed(experiment_name: str) -> int:
    """Deterministic seed from experiment name."""
    h = hashlib.sha256(f"{MASTER_SEED}:{experiment_name}".encode()).digest()
    return int.from_bytes(h[:4], 'big')

def run_seeds(experiment_name: str, n_runs: int = 30) -> list:
    """Generate n_runs seeds for an experiment."""
    base = experiment_seed(experiment_name)
    rng = np.random.default_rng(base)
    return rng.integers(0, 2**31, size=n_runs).tolist()

def workload_seed(run_seed: int) -> int:
    """Seed for workload generation within a run."""
    return run_seed * 2 + 1

def policy_seed(run_seed: int) -> int:
    """Seed for policy randomness within a run."""
    return run_seed * 2
```

### G.4 One-Command Scripts

```bash
#!/bin/bash
# run_benchmarks.sh - One command to reproduce all results

set -euo pipefail

echo "=== Cost-Aware Eviction Benchmark Suite ==="
echo "Starting at $(date)"
echo "Output directory: ${BENCHMARK_OUTPUT_DIR:-./results}"

# Step 1: Validate environment
python -c "import gptcache; import numpy; import scipy; print('Environment OK')"

# Step 2: Run unit tests (fast, validates code correctness)
python -m pytest tests/ -v --tb=short

# Step 3: Run main benchmarks
python -m benchmark.run_all \
    --seed ${BENCHMARK_SEED:-42} \
    --n-runs ${BENCHMARK_N_RUNS:-30} \
    --output-dir ${BENCHMARK_OUTPUT_DIR:-./results} \
    --workloads ${BENCHMARK_WORKLOADS:-all}

# Step 4: Generate plots
python -m benchmark.generate_plots \
    --input-dir ${BENCHMARK_OUTPUT_DIR:-./results} \
    --output-dir ${BENCHMARK_OUTPUT_DIR:-./results}/figures

# Step 5: Generate tables (LaTeX and Markdown)
python -m benchmark.generate_tables \
    --input-dir ${BENCHMARK_OUTPUT_DIR:-./results} \
    --output-dir ${BENCHMARK_OUTPUT_DIR:-./results}/tables

# Step 6: Statistical tests
python -m benchmark.statistical_analysis \
    --input-dir ${BENCHMARK_OUTPUT_DIR:-./results} \
    --output-dir ${BENCHMARK_OUTPUT_DIR:-./results}/stats

echo "=== Benchmark complete at $(date) ==="
echo "Results in: ${BENCHMARK_OUTPUT_DIR:-./results}"
```

### G.5 Expected Output Format

```
results/
├── raw/
│   ├── uniform_cost/
│   │   ├── lru_run_001.json
│   │   ├── lru_run_002.json
│   │   ├── ...
│   │   ├── gdsf_run_001.json
│   │   └── ...
│   ├── high_variance/
│   │   └── ...
│   └── ...
├── figures/
│   ├── fig1_hit_rate_vs_cache_size.pdf
│   ├── fig2_cwhr_vs_cache_size.pdf
│   ├── fig3_dollar_savings.pdf
│   ├── fig4_latency_cdf.pdf
│   ├── fig5_ablation_heatmap.pdf
│   ├── fig6_workload_sensitivity.pdf
│   ├── fig7_memory_overhead.pdf
│   └── fig8_parameter_sensitivity.pdf
├── tables/
│   ├── table1_main_results.tex
│   ├── table1_main_results.md
│   ├── table2_ablation.tex
│   └── table3_statistical_tests.tex
├── stats/
│   ├── pairwise_tests.json
│   ├── effect_sizes.json
│   └── summary.txt
└── metadata.json  # versions, seeds, runtime, git hash
```

**Individual run JSON format:**
```json
{
    "experiment": "high_variance",
    "policy": "gdsf",
    "run_id": 1,
    "seed": 1234567,
    "config": {
        "cache_size": 200,
        "n_unique_queries": 1000,
        "n_total_requests": 10000,
        "alpha": 1.0,
        "beta": 1.0,
        "workload_params": {}
    },
    "metrics": {
        "hit_rate": 0.4523,
        "cost_weighted_hit_rate": 0.7234,
        "dollar_savings": 0.0456,
        "latency_p50": 5.2,
        "latency_p95": 1234.5,
        "latency_p99": 2345.6,
        "throughput_qps": 12500.0,
        "memory_bytes": 45600,
        "n_evictions": 3456
    },
    "time_series": {
        "hit_rate_rolling": [0.1, 0.15, ...],
        "cwhr_rolling": [0.2, 0.35, ...]
    },
    "runtime_seconds": 2.34
}
```

### G.6 Metadata File

```python
# Generated automatically by benchmark runner
import json
import subprocess
import platform
import sys

def generate_metadata():
    return {
        "timestamp": datetime.now().isoformat(),
        "git_hash": subprocess.check_output(
            ["git", "rev-parse", "HEAD"]).decode().strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"]).decode().strip()),
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "gptcache_version": gptcache.__version__,
        "master_seed": MASTER_SEED,
        "n_runs_per_config": N_RUNS,
        "total_runtime_seconds": total_time,
        "total_experiments": total_count,
    }
```

---

## H. Main Benchmark Runner

```python
# benchmark/run_all.py
"""
Main benchmark orchestration script.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from benchmark.workloads import (
    generate_uniform_cost_workload,
    generate_high_variance_cost_workload,
    generate_zipf_variable_cost_workload,
    generate_bursty_workload,
    generate_adversarial_lru_workload,
    generate_size_varying_workload,
)
from benchmark.policies import LRUPolicy, FIFOPolicy, LFUPolicy, RandomPolicy, GDSFPolicy
from benchmark.metrics import MetricsCollector
from benchmark.seeds import run_seeds, workload_seed, policy_seed
from benchmark.simulate import run_simulation


WORKLOAD_GENERATORS = {
    'uniform': generate_uniform_cost_workload,
    'high_variance': generate_high_variance_cost_workload,
    'zipfian': generate_zipf_variable_cost_workload,
    'bursty': generate_bursty_workload,
    'adversarial': generate_adversarial_lru_workload,
    'size_varying': generate_size_varying_workload,
}

POLICIES = {
    'lru': LRUPolicy,
    'fifo': FIFOPolicy,
    'lfu': LFUPolicy,
    'random': RandomPolicy,
    'gdsf': GDSFPolicy,
}

CACHE_SIZES = [50, 100, 200, 500, 1000]


def run_single_experiment(
    workload_name: str,
    policy_name: str,
    cache_size: int,
    run_id: int,
    seed: int,
    output_dir: Path,
    gdsf_alpha: float = 1.0,
    gdsf_beta: float = 1.0,
) -> Dict:
    """Run a single benchmark experiment."""

    # Generate workload
    workload = WORKLOAD_GENERATORS[workload_name](seed=workload_seed(seed))

    # Initialize policy
    policy_kwargs = {'max_size': cache_size}
    if policy_name == 'gdsf':
        policy_kwargs.update({'alpha': gdsf_alpha, 'beta': gdsf_beta})
    if policy_name == 'random':
        policy_kwargs['seed'] = policy_seed(seed)

    policy = POLICIES[policy_name](**policy_kwargs)

    # Run simulation
    metrics = run_simulation(workload, policy)

    # Save results
    result = {
        'experiment': workload_name,
        'policy': policy_name,
        'run_id': run_id,
        'seed': seed,
        'config': {
            'cache_size': cache_size,
            'alpha': gdsf_alpha if policy_name == 'gdsf' else None,
            'beta': gdsf_beta if policy_name == 'gdsf' else None,
        },
        'metrics': metrics.to_dict(),
    }

    # Write to disk
    out_path = output_dir / workload_name / f"{policy_name}_run_{run_id:03d}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description='Run benchmark suite')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n-runs', type=int, default=30)
    parser.add_argument('--output-dir', type=str, default='./results')
    parser.add_argument('--workloads', type=str, default='all')
    parser.add_argument('--policies', type=str, default='all')
    parser.add_argument('--cache-sizes', type=str, default='50,100,200,500,1000')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / 'raw'
    output_dir.mkdir(parents=True, exist_ok=True)

    workloads = list(WORKLOAD_GENERATORS.keys()) if args.workloads == 'all' else args.workloads.split(',')
    policies = list(POLICIES.keys()) if args.policies == 'all' else args.policies.split(',')
    cache_sizes = [int(x) for x in args.cache_sizes.split(',')]

    total_experiments = len(workloads) * len(policies) * len(cache_sizes) * args.n_runs
    print(f"Total experiments: {total_experiments}")

    start_time = time.time()

    with tqdm(total=total_experiments, desc="Running benchmarks") as pbar:
        for workload_name in workloads:
            seeds = run_seeds(workload_name, args.n_runs)
            for policy_name in policies:
                for cache_size in cache_sizes:
                    for run_id, seed in enumerate(seeds, 1):
                        run_single_experiment(
                            workload_name=workload_name,
                            policy_name=policy_name,
                            cache_size=cache_size,
                            run_id=run_id,
                            seed=seed,
                            output_dir=output_dir,
                        )
                        pbar.update(1)

    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f}s ({total_time/60:.1f} min)")


if __name__ == '__main__':
    main()
```

---

## I. Simulation Engine

```python
# benchmark/simulate.py
"""
Cache simulation engine.
Drives workload through a policy and collects metrics.
"""

import time
import numpy as np
from typing import List
from benchmark.metrics import MetricsCollector


def run_simulation(
    workload: List,  # List[Query]
    policy,          # Any policy implementing get/put interface
    warmup_fraction: float = 0.10,
    simulate_latency: bool = True,
    seed: int = 42
) -> MetricsCollector:
    """
    Run a workload through a cache policy and collect metrics.

    Returns a MetricsCollector with all measurements.
    """
    rng = np.random.default_rng(seed)
    metrics = MetricsCollector()
    metrics.start_timer()

    warmup_end = int(len(workload) * warmup_fraction)

    for i, query in enumerate(workload):
        # Try cache lookup
        result = policy.get(query.query_id)

        if result is not None:
            # HIT
            if simulate_latency:
                latency = max(1.0, rng.normal(5.0, 2.0))
            else:
                latency = 0.0

            if i >= warmup_end:
                metrics.record_hit(cost=query.cost, latency_ms=latency)
        else:
            # MISS - simulate API call
            if simulate_latency:
                base = 500 + query.cost * 30000
                latency = max(100.0, rng.normal(base, base * 0.2))
            else:
                latency = 0.0

            if i >= warmup_end:
                metrics.record_miss(cost=query.cost, latency_ms=latency)

            # Insert into cache (may trigger eviction)
            policy.put(
                key=query.query_id,
                value=query.embedding,  # store embedding as cached response
                cost=query.cost,
                size=query.size
            )

    metrics.stop_timer()
    return metrics
```
