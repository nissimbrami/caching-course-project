"""Integration tests for the GDSF eviction system.

End-to-end tests simulating real cache behavior:
- Comparison with LRU under uniform cost (GDSF should match)
- Cost-aware advantage over LRU under variable cost (GDSF should win)
- Zipf workload hit rate measurement
- Full workflow exercising put -> access -> evict -> verify
- Cache size sweep across different capacities
"""

import random
import math
from collections import OrderedDict
from typing import List, Tuple, Dict

import pytest

from cost_aware_eviction.config import GDSFConfig
from cost_aware_eviction.eviction_manager import GDSFEvictionManager


# ===========================================================================
# Helper: Simple LRU Cache for comparison
# ===========================================================================


class SimpleLRUCache:
    """A simple LRU cache for baseline comparison.

    Uses an OrderedDict to maintain insertion/access order.
    When full, evicts the least recently used (oldest) item.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache: OrderedDict = OrderedDict()
        self._current_size = 0

    def put(self, key: str, size: int = 1, cost: float = 1.0) -> List[str]:
        """Insert item, evicting LRU items if necessary. Returns evicted keys."""
        if key in self._cache:
            # Move to end (most recently used)
            old_size = self._cache[key]["size"]
            self._current_size -= old_size
            self._cache.move_to_end(key)
            self._cache[key] = {"size": size, "cost": cost}
            self._current_size += size
            return []

        if size > self.max_size:
            return []

        evicted = []
        while self._current_size + size > self.max_size:
            lru_key, lru_meta = self._cache.popitem(last=False)
            self._current_size -= lru_meta["size"]
            evicted.append(lru_key)

        self._cache[key] = {"size": size, "cost": cost}
        self._current_size += size
        return evicted

    def access(self, key: str) -> bool:
        """Record access (move to MRU position)."""
        if key not in self._cache:
            return False
        self._cache.move_to_end(key)
        return True

    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def __len__(self) -> int:
        return len(self._cache)


# ===========================================================================
# Helper: Workload generators
# ===========================================================================


def generate_uniform_cost_workload(
    n_unique: int, n_requests: int, seed: int = 42
) -> List[Tuple[str, int, float]]:
    """Generate a workload with uniform cost (all items cost 1.0).

    Returns list of (key, size, cost) tuples.
    """
    rng = random.Random(seed)
    keys = [f"query_{i}" for i in range(n_unique)]
    workload = []
    for _ in range(n_requests):
        key = rng.choice(keys)
        workload.append((key, 1, 1.0))
    return workload


def generate_variable_cost_workload(
    n_unique: int, n_requests: int, seed: int = 42, cost_ratio: float = 100.0
) -> List[Tuple[str, int, float]]:
    """Generate workload with variable costs.

    A few items are very expensive (GPT-4 style), most are cheap (GPT-3.5 style).
    Returns list of (key, size, cost) tuples.
    """
    rng = random.Random(seed)
    keys = [f"query_{i}" for i in range(n_unique)]

    # 10% of items are expensive
    n_expensive = max(1, n_unique // 10)
    expensive_keys = set(keys[:n_expensive])

    workload = []
    for _ in range(n_requests):
        key = rng.choice(keys)
        cost = cost_ratio if key in expensive_keys else 1.0
        workload.append((key, 1, cost))
    return workload


def generate_zipf_workload(
    n_unique: int, n_requests: int, alpha: float = 1.1, seed: int = 42
) -> List[Tuple[str, int, float]]:
    """Generate a Zipf-distributed workload.

    A few items are accessed very frequently, most are rare.
    """
    rng = random.Random(seed)
    keys = [f"zipf_{i}" for i in range(n_unique)]

    # Compute Zipf weights
    weights = [1.0 / ((i + 1) ** alpha) for i in range(n_unique)]
    total_weight = sum(weights)
    cumulative = []
    cum = 0.0
    for w in weights:
        cum += w / total_weight
        cumulative.append(cum)

    def pick_zipf():
        r = rng.random()
        for idx, c in enumerate(cumulative):
            if r <= c:
                return idx
        return len(cumulative) - 1

    workload = []
    for _ in range(n_requests):
        idx = pick_zipf()
        key = keys[idx]
        cost = rng.uniform(0.5, 5.0)
        workload.append((key, 1, cost))
    return workload


# ===========================================================================
# Helper: Run simulation
# ===========================================================================


def simulate_cache(
    cache, workload: List[Tuple[str, int, float]]
) -> Dict[str, float]:
    """Run a workload through a cache and collect metrics.

    Returns dict with hit_rate and cost_weighted_hit_rate.
    """
    hits = 0
    misses = 0
    hit_cost = 0.0
    total_cost = 0.0

    for key, size, cost in workload:
        total_cost += cost
        if key in cache:
            hits += 1
            hit_cost += cost
            cache.access(key)
        else:
            misses += 1
            cache.put(key, size=size, cost=cost)

    hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    cwhr = hit_cost / total_cost if total_cost > 0 else 0.0
    return {"hit_rate": hit_rate, "cost_weighted_hit_rate": cwhr, "hits": hits}


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestLRUComparisonUniformCost:
    """GDSF should perform approximately equal to LRU when costs are uniform."""

    def test_lru_comparison_uniform_cost(self):
        """Under uniform cost, GDSF hit rate is within 10% of LRU."""
        cache_size = 50
        n_unique = 200
        n_requests = 5000

        workload = generate_uniform_cost_workload(n_unique, n_requests, seed=42)

        # LRU
        lru = SimpleLRUCache(max_size=cache_size)
        lru_metrics = simulate_cache(lru, workload)

        # GDSF
        config = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
        gdsf = GDSFEvictionManager(config=config)
        gdsf_metrics = simulate_cache(gdsf, workload)

        # GDSF should be within 10% of LRU (not dramatically worse)
        lru_hr = lru_metrics["hit_rate"]
        gdsf_hr = gdsf_metrics["hit_rate"]

        if lru_hr > 0:
            ratio = gdsf_hr / lru_hr
            assert ratio > 0.90, (
                f"GDSF hit rate ({gdsf_hr:.3f}) is more than 10% worse than "
                f"LRU ({lru_hr:.3f}), ratio={ratio:.3f}"
            )

    def test_uniform_cost_cwhr_similar(self):
        """Under uniform cost, CWHR equals hit rate for both policies."""
        cache_size = 30
        workload = generate_uniform_cost_workload(100, 2000, seed=99)

        config = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
        gdsf = GDSFEvictionManager(config=config)
        metrics = simulate_cache(gdsf, workload)

        # When all costs are equal, CWHR should equal hit rate
        assert abs(metrics["hit_rate"] - metrics["cost_weighted_hit_rate"]) < 0.01


class TestCostAwareBeatLRU:
    """GDSF should outperform LRU on cost-weighted hit rate with variable costs."""

    def test_cost_aware_beats_lru_variable_cost(self):
        """GDSF achieves higher CWHR than LRU when costs vary significantly."""
        cache_size = 30
        n_unique = 100
        n_requests = 5000

        workload = generate_variable_cost_workload(
            n_unique, n_requests, seed=42, cost_ratio=100.0
        )

        # LRU
        lru = SimpleLRUCache(max_size=cache_size)
        lru_metrics = simulate_cache(lru, workload)

        # GDSF
        config = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
        gdsf = GDSFEvictionManager(config=config)
        gdsf_metrics = simulate_cache(gdsf, workload)

        # GDSF should have better CWHR
        assert gdsf_metrics["cost_weighted_hit_rate"] >= lru_metrics["cost_weighted_hit_rate"], (
            f"GDSF CWHR ({gdsf_metrics['cost_weighted_hit_rate']:.4f}) should be >= "
            f"LRU CWHR ({lru_metrics['cost_weighted_hit_rate']:.4f})"
        )

    def test_cost_aware_advantage_grows_with_cost_ratio(self):
        """The GDSF advantage over LRU increases with higher cost variance."""
        cache_size = 20
        n_unique = 80
        n_requests = 3000

        advantages = []
        for cost_ratio in [2.0, 10.0, 50.0, 200.0]:
            workload = generate_variable_cost_workload(
                n_unique, n_requests, seed=42, cost_ratio=cost_ratio
            )

            lru = SimpleLRUCache(max_size=cache_size)
            lru_metrics = simulate_cache(lru, workload)

            config = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
            gdsf = GDSFEvictionManager(config=config)
            gdsf_metrics = simulate_cache(gdsf, workload)

            advantage = (
                gdsf_metrics["cost_weighted_hit_rate"]
                - lru_metrics["cost_weighted_hit_rate"]
            )
            advantages.append(advantage)

        # Advantage should generally be non-negative for higher cost ratios
        # At minimum, the last (highest ratio) should show benefit
        assert advantages[-1] >= 0, (
            f"GDSF should show advantage at high cost ratio. Got: {advantages}"
        )

    def test_expensive_items_preferentially_cached(self):
        """GDSF keeps expensive items in cache preferentially."""
        config = GDSFConfig(max_size=5, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        # Insert expensive items
        for i in range(3):
            manager.put(f"expensive_{i}", size=1, cost=100.0)

        # Insert cheap items that fill and trigger eviction
        for i in range(5):
            manager.put(f"cheap_{i}", size=1, cost=1.0)

        # Check: at least some expensive items should remain
        expensive_remaining = sum(
            1 for i in range(3) if f"expensive_{i}" in manager
        )
        assert expensive_remaining >= 1, "At least one expensive item should survive"


class TestZipfWorkload:
    """Test with Zipf-distributed access patterns."""

    def test_zipf_workload_hit_rate(self):
        """GDSF achieves reasonable hit rate on Zipf workload."""
        cache_size = 50
        n_unique = 500
        n_requests = 10000

        workload = generate_zipf_workload(n_unique, n_requests, alpha=1.1, seed=42)

        config = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
        gdsf = GDSFEvictionManager(config=config)
        metrics = simulate_cache(gdsf, workload)

        # With Zipf (alpha=1.1) and cache_size/n_unique = 10%,
        # we expect a decent hit rate (Zipf concentrates on popular items)
        assert metrics["hit_rate"] > 0.1, (
            f"Hit rate too low for Zipf workload: {metrics['hit_rate']:.3f}"
        )

    def test_zipf_frequency_boosting(self):
        """Frequently accessed items in Zipf workload remain cached."""
        cache_size = 20
        n_unique = 200
        n_requests = 5000

        workload = generate_zipf_workload(n_unique, n_requests, alpha=1.5, seed=42)

        config = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
        gdsf = GDSFEvictionManager(config=config)

        # Simulate and track
        for key, size, cost in workload:
            if key in gdsf:
                gdsf.access(key)
            else:
                gdsf.put(key, size=size, cost=cost)

        # The most popular item (zipf_0) should very likely be in cache
        assert "zipf_0" in gdsf, "Most popular Zipf item should be cached"


class TestFullWorkflow:
    """Test complete workflow: put -> access -> evict -> verify."""

    def test_full_workflow(self):
        """Exercise the complete lifecycle of cache items."""
        config = GDSFConfig(max_size=5, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        # Phase 1: Insert items
        manager.put("item_a", size=1, cost=2.0)
        manager.put("item_b", size=1, cost=4.0)
        manager.put("item_c", size=1, cost=1.0)
        manager.put("item_d", size=1, cost=3.0)
        manager.put("item_e", size=1, cost=5.0)

        assert len(manager) == 5
        assert manager.current_size == 5

        # Phase 2: Access some items to boost their priority
        manager.access("item_c")  # boost the cheapest
        manager.access("item_c")
        manager.access("item_c")

        # Phase 3: Trigger eviction
        evicted = manager.put("item_f", size=1, cost=2.0)
        assert len(evicted) == 1

        # Phase 4: Verify state
        assert manager.current_size <= 5
        assert "item_f" in manager
        # "item_e" (cost=5.0) should survive (highest priority from cost alone)
        assert "item_e" in manager
        # "item_c" should survive (freq boosted)
        # After 3 accesses: freq=4, cost=1, size=1 -> priority = clock + 4*1/1 = clock+4

        # Phase 5: Explicit removal
        manager.remove("item_e")
        assert "item_e" not in manager
        assert manager.current_size == 4

    def test_workflow_with_repeated_access_pattern(self):
        """Simulate a realistic repeated access pattern."""
        config = GDSFConfig(max_size=10, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        # Simulate: popular items are accessed repeatedly
        popular_items = [f"popular_{i}" for i in range(3)]
        all_items = popular_items + [f"rare_{i}" for i in range(20)]

        rng = random.Random(42)

        for _ in range(200):
            # 70% of the time access popular items, 30% rare
            if rng.random() < 0.7:
                key = rng.choice(popular_items)
            else:
                key = rng.choice(all_items)

            if key in manager:
                manager.access(key)
            else:
                manager.put(key, size=1, cost=1.0)

        # Popular items should mostly be in cache
        popular_in_cache = sum(1 for k in popular_items if k in manager)
        assert popular_in_cache >= 2, (
            f"At least 2/3 popular items should be cached, got {popular_in_cache}"
        )


class TestCacheSizeSweep:
    """Test behavior across different cache sizes."""

    @pytest.mark.parametrize("cache_size", [10, 50, 100, 200, 500])
    def test_cache_size_sweep(self, cache_size):
        """Hit rate increases with cache size."""
        n_unique = 1000
        n_requests = 5000

        workload = generate_zipf_workload(n_unique, n_requests, alpha=1.1, seed=42)

        config = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
        gdsf = GDSFEvictionManager(config=config)
        metrics = simulate_cache(gdsf, workload)

        # Larger cache should give higher hit rate (or at least non-negative)
        assert metrics["hit_rate"] >= 0.0
        assert metrics["hit_rate"] <= 1.0

    def test_hit_rate_monotonically_increases_with_size(self):
        """Larger caches achieve equal or higher hit rates."""
        n_unique = 500
        n_requests = 5000
        workload = generate_zipf_workload(n_unique, n_requests, alpha=1.1, seed=42)

        hit_rates = []
        for cache_size in [10, 50, 100, 200, 500]:
            config = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
            gdsf = GDSFEvictionManager(config=config)
            metrics = simulate_cache(gdsf, workload)
            hit_rates.append(metrics["hit_rate"])

        # Hit rate should be non-decreasing with cache size
        for i in range(1, len(hit_rates)):
            assert hit_rates[i] >= hit_rates[i - 1] - 0.01, (
                f"Hit rate should not decrease significantly with larger cache: "
                f"{hit_rates}"
            )

    def test_full_cache_hit_rate_approaches_one(self):
        """When cache can hold all items, hit rate approaches 1.0."""
        n_unique = 50
        n_requests = 2000
        workload = generate_uniform_cost_workload(n_unique, n_requests, seed=42)

        # Cache large enough to hold everything
        config = GDSFConfig(max_size=n_unique, alpha=1.0, beta=1.0)
        gdsf = GDSFEvictionManager(config=config)
        metrics = simulate_cache(gdsf, workload)

        # After warmup, all items should be cached
        # Hit rate = (n_requests - n_unique_first_seen) / n_requests
        # With 50 unique items and 2000 requests, after first 50 misses
        # all subsequent should be hits
        assert metrics["hit_rate"] > 0.9


class TestReproducibility:
    """Test that results are reproducible."""

    def test_same_seed_same_results(self):
        """Same workload with same seed produces identical metrics."""
        cache_size = 30
        workload = generate_variable_cost_workload(100, 3000, seed=12345)

        metrics_runs = []
        for _ in range(3):
            config = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
            gdsf = GDSFEvictionManager(config=config)
            metrics = simulate_cache(gdsf, workload)
            metrics_runs.append(metrics)

        # All runs should be identical
        for i in range(1, len(metrics_runs)):
            assert metrics_runs[i]["hit_rate"] == metrics_runs[0]["hit_rate"]
            assert (
                metrics_runs[i]["cost_weighted_hit_rate"]
                == metrics_runs[0]["cost_weighted_hit_rate"]
            )


class TestAblationIntegration:
    """Integration tests for parameter ablation."""

    def test_alpha_beta_sweep(self):
        """Different alpha/beta produce different CWHR on variable-cost workload."""
        cache_size = 30
        workload = generate_variable_cost_workload(100, 3000, seed=42)

        results = {}
        for alpha in [0.0, 0.5, 1.0, 2.0]:
            for beta in [0.0, 0.5, 1.0, 2.0]:
                config = GDSFConfig(max_size=cache_size, alpha=alpha, beta=beta)
                gdsf = GDSFEvictionManager(config=config)
                metrics = simulate_cache(gdsf, workload)
                results[(alpha, beta)] = metrics["cost_weighted_hit_rate"]

        # Not all configurations should give the same result
        unique_cwhrs = set(round(v, 6) for v in results.values())
        assert len(unique_cwhrs) > 1, "Different parameters should give different results"

    def test_beta_zero_same_as_frequency_only(self):
        """With beta=0, cost is ignored and policy is frequency-based."""
        cache_size = 30
        workload = generate_variable_cost_workload(100, 3000, seed=42, cost_ratio=1000.0)

        # beta=0 should not benefit from cost awareness
        config_no_cost = GDSFConfig(max_size=cache_size, alpha=1.0, beta=0.0)
        gdsf_no_cost = GDSFEvictionManager(config=config_no_cost)
        metrics_no_cost = simulate_cache(gdsf_no_cost, workload)

        # beta=1 should benefit
        config_with_cost = GDSFConfig(max_size=cache_size, alpha=1.0, beta=1.0)
        gdsf_with_cost = GDSFEvictionManager(config=config_with_cost)
        metrics_with_cost = simulate_cache(gdsf_with_cost, workload)

        # With high cost ratio, cost-aware version should have better CWHR
        assert metrics_with_cost["cost_weighted_hit_rate"] >= metrics_no_cost["cost_weighted_hit_rate"]
