"""
Workload generators for cache eviction policy benchmarks.

Each generator produces a list of query dicts simulating different access patterns
found in real-world LLM caching scenarios. Queries carry cost, size, and timing
metadata used by cost-aware eviction policies like GDSF.
"""

from typing import List, Dict, Any
import numpy as np


def _make_query(
    query_id: int,
    cost: float,
    size: int,
    timestamp: float,
    model_name: str,
) -> Dict[str, Any]:
    """Create a standardized query dict."""
    return {
        "query_id": query_id,
        "cost": cost,
        "size": size,
        "timestamp": timestamp,
        "model_name": model_name,
    }


def generate_uniform_cost_workload(
    n_queries: int = 10000,
    n_unique: int = 1000,
    cost: float = 1.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a workload where all queries have the same cost.

    This is the baseline workload where cost-aware policies should behave
    similarly to frequency-based policies since cost provides no signal.

    Args:
        n_queries: Total number of queries to generate.
        n_unique: Number of unique query IDs in the workload.
        cost: Uniform cost assigned to every query.
        seed: Random seed for reproducibility.

    Returns:
        List of query dicts with uniform cost.
    """
    rng = np.random.default_rng(seed)
    queries: List[Dict[str, Any]] = []

    for i in range(n_queries):
        qid = int(rng.integers(0, n_unique))
        queries.append(
            _make_query(
                query_id=qid,
                cost=cost,
                size=512,  # uniform size
                timestamp=float(i),
                model_name="gpt-3.5-turbo",
            )
        )

    return queries


def generate_high_variance_cost_workload(
    n_queries: int = 10000,
    n_unique: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a trimodal cost workload simulating mixed model usage.

    Distribution:
        - 60% cheap queries (GPT-3.5-turbo, cost ~ 0.002)
        - 30% medium queries (GPT-4, cost ~ 0.06)
        - 10% expensive queries (GPT-4-32k, cost ~ 0.12)

    This workload tests whether cost-aware policies correctly prioritize
    retention of expensive query results.

    Args:
        n_queries: Total number of queries to generate.
        n_unique: Number of unique query IDs in the workload.
        seed: Random seed for reproducibility.

    Returns:
        List of query dicts with trimodal cost distribution.
    """
    rng = np.random.default_rng(seed)
    queries: List[Dict[str, Any]] = []

    # Assign each unique query a tier
    tier_assignments = rng.choice(
        ["cheap", "medium", "expensive"],
        size=n_unique,
        p=[0.6, 0.3, 0.1],
    )

    tier_config = {
        "cheap": {"cost_mean": 0.002, "cost_std": 0.0005, "model": "gpt-3.5-turbo", "size_mean": 256},
        "medium": {"cost_mean": 0.06, "cost_std": 0.01, "model": "gpt-4", "size_mean": 512},
        "expensive": {"cost_mean": 0.12, "cost_std": 0.02, "model": "gpt-4-32k", "size_mean": 1024},
    }

    for i in range(n_queries):
        qid = int(rng.integers(0, n_unique))
        tier = tier_assignments[qid]
        config = tier_config[tier]

        cost = max(0.0001, rng.normal(config["cost_mean"], config["cost_std"]))
        size = max(64, int(rng.normal(config["size_mean"], config["size_mean"] * 0.2)))

        queries.append(
            _make_query(
                query_id=qid,
                cost=float(cost),
                size=size,
                timestamp=float(i),
                model_name=config["model"],
            )
        )

    return queries


def generate_zipf_variable_cost_workload(
    n_queries: int = 10000,
    n_unique: int = 1000,
    alpha: float = 1.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a Zipfian popularity workload with negative cost-popularity correlation.

    Popular queries are cheap (e.g., simple lookups served by GPT-3.5),
    while rare queries are expensive (e.g., complex reasoning by GPT-4-32k).
    This tests whether policies keep expensive but rare items over cheap popular ones.

    Args:
        n_queries: Total number of queries to generate.
        n_unique: Number of unique query IDs in the workload.
        alpha: Zipf distribution parameter (higher = more skewed).
        seed: Random seed for reproducibility.

    Returns:
        List of query dicts with Zipfian access and inverse cost correlation.
    """
    rng = np.random.default_rng(seed)
    queries: List[Dict[str, Any]] = []

    # Assign costs inversely proportional to popularity rank
    # Rank 1 (most popular) gets lowest cost, rank n_unique gets highest cost
    ranks = np.arange(1, n_unique + 1)
    base_costs = 0.002 + (0.12 - 0.002) * (ranks - 1) / (n_unique - 1)
    # Add small noise
    costs = base_costs + rng.normal(0, 0.001, size=n_unique)
    costs = np.clip(costs, 0.001, 0.2)

    # Sizes also increase with rank (rare queries have longer responses)
    sizes = (256 + (4096 - 256) * (ranks - 1) / (n_unique - 1)).astype(int)

    # Model assignment based on cost tier
    models = []
    for c in costs:
        if c < 0.01:
            models.append("gpt-3.5-turbo")
        elif c < 0.08:
            models.append("gpt-4")
        else:
            models.append("gpt-4-32k")

    # Generate Zipfian query IDs
    # numpy's zipf returns values >= 1, we clip to n_unique
    zipf_samples = rng.zipf(alpha + 1, size=n_queries)  # +1 because numpy uses s > 1
    query_ids = (zipf_samples - 1) % n_unique

    for i in range(n_queries):
        qid = int(query_ids[i])
        queries.append(
            _make_query(
                query_id=qid,
                cost=float(costs[qid]),
                size=int(sizes[qid]),
                timestamp=float(i),
                model_name=models[qid],
            )
        )

    return queries


def generate_bursty_workload(
    n_queries: int = 10000,
    n_unique: int = 500,
    burst_size: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a workload with bursty access patterns.

    Simulates scenarios where certain topics trend temporarily (e.g., news events)
    causing bursts of related queries, followed by periods of diverse access.

    Args:
        n_queries: Total number of queries to generate.
        n_unique: Number of unique query IDs in the workload.
        burst_size: Number of consecutive queries in a burst.
        seed: Random seed for reproducibility.

    Returns:
        List of query dicts with bursty access patterns.
    """
    rng = np.random.default_rng(seed)
    queries: List[Dict[str, Any]] = []

    # Assign random costs and sizes to each unique query
    costs = rng.choice(
        [0.002, 0.06, 0.12],
        size=n_unique,
        p=[0.5, 0.35, 0.15],
    )
    sizes = rng.integers(128, 4096, size=n_unique)
    model_map = {0.002: "gpt-3.5-turbo", 0.06: "gpt-4", 0.12: "gpt-4-32k"}
    models = [model_map[c] for c in costs]

    i = 0
    while i < n_queries:
        # Decide whether this segment is a burst or background
        if rng.random() < 0.3:  # 30% chance of a burst
            # Pick a small set of queries for the burst
            burst_queries = rng.integers(0, n_unique, size=min(5, n_unique))
            actual_burst_size = min(burst_size, n_queries - i)
            for _ in range(actual_burst_size):
                qid = int(rng.choice(burst_queries))
                queries.append(
                    _make_query(
                        query_id=qid,
                        cost=float(costs[qid]),
                        size=int(sizes[qid]),
                        timestamp=float(i),
                        model_name=models[qid],
                    )
                )
                i += 1
        else:
            # Background traffic: uniform random
            segment_size = min(rng.integers(10, 50), n_queries - i)
            for _ in range(segment_size):
                qid = int(rng.integers(0, n_unique))
                queries.append(
                    _make_query(
                        query_id=qid,
                        cost=float(costs[qid]),
                        size=int(sizes[qid]),
                        timestamp=float(i),
                        model_name=models[qid],
                    )
                )
                i += 1

    return queries[:n_queries]


def generate_adversarial_lru_workload(
    n_queries: int = 10000,
    cache_size: int = 100,
    n_expensive_recurring: int = 20,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a workload adversarial to LRU eviction.

    Pattern: A scan of unique queries (larger than cache) interleaved with
    expensive recurring queries. LRU will evict the expensive recurring queries
    during the scan, while cost-aware policies should retain them.

    Args:
        n_queries: Total number of queries to generate.
        cache_size: Simulated cache capacity (for sizing the scan).
        n_expensive_recurring: Number of expensive queries that recur regularly.
        seed: Random seed for reproducibility.

    Returns:
        List of query dicts adversarial to LRU.
    """
    rng = np.random.default_rng(seed)
    queries: List[Dict[str, Any]] = []

    # Expensive recurring queries: IDs 0 to n_expensive_recurring-1
    expensive_cost = 0.12
    expensive_size = 1024

    # Scan queries: IDs starting from n_expensive_recurring
    # These are one-shot queries that appear once and pollute the cache
    scan_base_id = n_expensive_recurring
    n_scan_queries = cache_size * 2  # Larger than cache to force evictions
    scan_cost = 0.002
    scan_size = 256

    i = 0
    scan_counter = 0
    while i < n_queries:
        # Phase 1: Access some expensive recurring queries
        n_expensive_access = min(rng.integers(3, 8), n_queries - i)
        for _ in range(n_expensive_access):
            if i >= n_queries:
                break
            qid = int(rng.integers(0, n_expensive_recurring))
            queries.append(
                _make_query(
                    query_id=qid,
                    cost=expensive_cost,
                    size=expensive_size,
                    timestamp=float(i),
                    model_name="gpt-4-32k",
                )
            )
            i += 1

        # Phase 2: Scan with cheap one-shot queries (pollutes LRU)
        scan_length = min(rng.integers(cache_size, cache_size * 2), n_queries - i)
        for _ in range(scan_length):
            if i >= n_queries:
                break
            qid = scan_base_id + (scan_counter % n_scan_queries)
            scan_counter += 1
            queries.append(
                _make_query(
                    query_id=qid,
                    cost=scan_cost,
                    size=scan_size,
                    timestamp=float(i),
                    model_name="gpt-3.5-turbo",
                )
            )
            i += 1

    return queries[:n_queries]


def generate_size_varying_workload(
    n_queries: int = 10000,
    n_unique: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generate a workload with highly variable response sizes.

    Sizes range from 50 bytes (short answers) to 50KB (long code/explanations).
    Cost is weakly correlated with size (longer responses cost more tokens).
    Tests whether size-aware eviction (the S in GDSF) helps.

    Args:
        n_queries: Total number of queries to generate.
        n_unique: Number of unique query IDs in the workload.
        seed: Random seed for reproducibility.

    Returns:
        List of query dicts with variable response sizes.
    """
    rng = np.random.default_rng(seed)
    queries: List[Dict[str, Any]] = []

    # Log-normal size distribution: most are small, some are very large
    log_sizes = rng.normal(loc=6.0, scale=1.5, size=n_unique)  # ln(size)
    sizes = np.clip(np.exp(log_sizes), 50, 50000).astype(int)

    # Cost weakly correlated with size (larger responses cost more)
    # cost ~ 0.002 + 0.003 * (size / 1000) + noise
    costs = 0.002 + 0.003 * (sizes / 1000.0) + rng.normal(0, 0.005, size=n_unique)
    costs = np.clip(costs, 0.001, 0.2)

    # Model assignment based on cost
    models = []
    for c in costs:
        if c < 0.01:
            models.append("gpt-3.5-turbo")
        elif c < 0.08:
            models.append("gpt-4")
        else:
            models.append("gpt-4-32k")

    for i in range(n_queries):
        qid = int(rng.integers(0, n_unique))
        queries.append(
            _make_query(
                query_id=qid,
                cost=float(costs[qid]),
                size=int(sizes[qid]),
                timestamp=float(i),
                model_name=models[qid],
            )
        )

    return queries


# Registry of all workloads for easy iteration
WORKLOAD_REGISTRY: Dict[str, Any] = {
    "uniform_cost": generate_uniform_cost_workload,
    "high_variance_cost": generate_high_variance_cost_workload,
    "zipf_variable_cost": generate_zipf_variable_cost_workload,
    "bursty": generate_bursty_workload,
    "adversarial_lru": generate_adversarial_lru_workload,
    "size_varying": generate_size_varying_workload,
}


def get_workload(name: str, **kwargs) -> List[Dict[str, Any]]:
    """Get a workload by name from the registry.

    Args:
        name: Name of the workload generator (key in WORKLOAD_REGISTRY).
        **kwargs: Keyword arguments passed to the generator function.

    Returns:
        List of query dicts from the specified workload.

    Raises:
        ValueError: If the workload name is not found in the registry.
    """
    if name not in WORKLOAD_REGISTRY:
        raise ValueError(
            f"Unknown workload '{name}'. Available: {list(WORKLOAD_REGISTRY.keys())}"
        )
    return WORKLOAD_REGISTRY[name](**kwargs)
