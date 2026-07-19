"""
Metrics collection and result aggregation for cache benchmarks.

Provides the BenchmarkResult dataclass for structured results and the
MetricsCollector class for incrementally recording hits/misses during
a benchmark run.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
import time
import statistics
import sys
import os
import threading

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


@dataclass
class BenchmarkResult:
    """Complete result of a single benchmark experiment.

    Captures both effectiveness metrics (hit rate, cost savings) and
    efficiency metrics (latency, throughput, memory).
    """

    policy_name: str
    workload_name: str
    cache_size: int
    hit_rate: float
    cost_weighted_hit_rate: float  # sum(cost for hits) / sum(cost for all)
    dollar_savings: float  # total cost of hits (money saved by caching)
    latency_p50: float  # median operation latency in ms
    latency_p95: float  # 95th percentile latency in ms
    latency_p99: float  # 99th percentile latency in ms
    throughput: float  # queries processed per second
    memory_overhead_bytes: int  # approximate memory used by the policy
    n_queries: int  # total number of queries processed
    n_evictions: int  # total evictions performed
    run_id: int  # identifier for this run (for multi-run experiments)
    seed: int  # random seed used for this run
    cpu_percent_mean: float = 0.0  # mean process CPU% sampled during run
    cpu_percent_p95: float = 0.0   # 95th percentile process CPU%
    rss_mean_mb: float = 0.0       # mean resident set size in MB during run
    rss_peak_mb: float = 0.0       # peak resident set size in MB during run
    gpu_utilization: str = "N/A (GPTCache eviction path is CPU-only)"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a flat dictionary suitable for CSV/JSON serialization."""
        return asdict(self)

    def to_csv_row(self) -> Dict[str, Any]:
        """Alias for to_dict, for CSV writer compatibility."""
        return self.to_dict()


class MetricsCollector:
    """Incrementally collects metrics during a benchmark run.

    Records individual hit/miss events with cost and latency, then
    computes aggregate statistics on demand.

    Usage:
        collector = MetricsCollector(policy_name="LRU", workload_name="zipf", ...)
        for query in workload:
            start = time.perf_counter()
            hit = policy.access(key)
            latency = (time.perf_counter() - start) * 1000
            if hit:
                collector.record_hit(query["cost"], latency)
            else:
                collector.record_miss(query["cost"], latency)
        result = collector.get_result()
    """

    def __init__(
        self,
        policy_name: str,
        workload_name: str,
        cache_size: int,
        run_id: int = 0,
        seed: int = 42,
    ) -> None:
        """Initialize the metrics collector.

        Args:
            policy_name: Name of the cache policy being benchmarked.
            workload_name: Name of the workload being used.
            cache_size: Maximum cache size in bytes.
            run_id: Identifier for this particular run.
            seed: Random seed used for this run.
        """
        self.policy_name = policy_name
        self.workload_name = workload_name
        self.cache_size = cache_size
        self.run_id = run_id
        self.seed = seed

        self._hits: int = 0
        self._misses: int = 0
        self._hit_costs: List[float] = []
        self._total_cost: float = 0.0
        self._latencies: List[float] = []
        self._n_evictions: int = 0
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._memory_overhead: int = 0
        # Resource sampling (CPU%, RSS). Populated by ResourceSampler if used.
        self._cpu_samples: List[float] = []
        self._rss_samples_mb: List[float] = []

    def start_timer(self) -> None:
        """Mark the beginning of the benchmark run."""
        self._start_time = time.perf_counter()

    def stop_timer(self) -> None:
        """Mark the end of the benchmark run."""
        self._end_time = time.perf_counter()

    def record_hit(self, query_cost: float, latency_ms: float) -> None:
        """Record a cache hit event.

        Args:
            query_cost: Dollar cost of the query that was served from cache.
            latency_ms: Time taken for the cache lookup in milliseconds.
        """
        self._hits += 1
        self._hit_costs.append(query_cost)
        self._total_cost += query_cost
        self._latencies.append(latency_ms)

    def record_miss(self, query_cost: float, latency_ms: float) -> None:
        """Record a cache miss event.

        Args:
            query_cost: Dollar cost of the query that had to be recomputed.
            latency_ms: Time taken for the cache lookup + insertion in milliseconds.
        """
        self._misses += 1
        self._total_cost += query_cost
        self._latencies.append(latency_ms)

    def record_evictions(self, n: int) -> None:
        """Record that n evictions occurred.

        Args:
            n: Number of items evicted.
        """
        self._n_evictions += n

    def set_memory_overhead(self, n_bytes: int) -> None:
        """Set the measured memory overhead of the policy.

        Args:
            n_bytes: Approximate memory usage in bytes.
        """
        self._memory_overhead = n_bytes

    def record_resource_sample(self, cpu_percent: float, rss_mb: float) -> None:
        """Record a resource-utilization sample (CPU% and RSS in MB).

        Called by ResourceSampler at fixed intervals during the benchmark run.
        """
        self._cpu_samples.append(cpu_percent)
        self._rss_samples_mb.append(rss_mb)

    @property
    def n_queries(self) -> int:
        """Total number of queries processed so far."""
        return self._hits + self._misses

    def get_result(self) -> BenchmarkResult:
        """Compute and return the final benchmark result.

        Returns:
            BenchmarkResult with all computed metrics.

        Raises:
            ValueError: If no queries have been recorded.
        """
        total = self._hits + self._misses
        if total == 0:
            raise ValueError("No queries recorded. Cannot compute metrics.")

        # Hit rate
        hit_rate = self._hits / total

        # Cost-weighted hit rate
        hit_cost_sum = sum(self._hit_costs)
        cost_weighted_hit_rate = hit_cost_sum / self._total_cost if self._total_cost > 0 else 0.0

        # Dollar savings (total cost saved by cache hits)
        dollar_savings = hit_cost_sum

        # Latency percentiles
        sorted_latencies = sorted(self._latencies) if self._latencies else [0.0]
        latency_p50 = self._percentile(sorted_latencies, 50)
        latency_p95 = self._percentile(sorted_latencies, 95)
        latency_p99 = self._percentile(sorted_latencies, 99)

        # Throughput
        if self._start_time is not None and self._end_time is not None:
            elapsed = self._end_time - self._start_time
            throughput = total / elapsed if elapsed > 0 else 0.0
        else:
            # Estimate from latencies
            total_latency_s = sum(self._latencies) / 1000.0
            throughput = total / total_latency_s if total_latency_s > 0 else 0.0

        return BenchmarkResult(
            policy_name=self.policy_name,
            workload_name=self.workload_name,
            cache_size=self.cache_size,
            hit_rate=hit_rate,
            cost_weighted_hit_rate=cost_weighted_hit_rate,
            dollar_savings=dollar_savings,
            latency_p50=latency_p50,
            latency_p95=latency_p95,
            latency_p99=latency_p99,
            throughput=throughput,
            memory_overhead_bytes=self._memory_overhead,
            n_queries=total,
            n_evictions=self._n_evictions,
            run_id=self.run_id,
            seed=self.seed,
            cpu_percent_mean=(sum(self._cpu_samples) / len(self._cpu_samples)) if self._cpu_samples else 0.0,
            cpu_percent_p95=self._percentile(sorted(self._cpu_samples), 95) if self._cpu_samples else 0.0,
            rss_mean_mb=(sum(self._rss_samples_mb) / len(self._rss_samples_mb)) if self._rss_samples_mb else 0.0,
            rss_peak_mb=max(self._rss_samples_mb) if self._rss_samples_mb else 0.0,
        )

    def to_csv_row(self) -> Dict[str, Any]:
        """Get the result as a CSV-compatible dict.

        Returns:
            Dictionary with all metric fields.
        """
        return self.get_result().to_csv_row()

    @staticmethod
    def _percentile(sorted_data: List[float], p: float) -> float:
        """Compute the p-th percentile of sorted data.

        Args:
            sorted_data: Pre-sorted list of values.
            p: Percentile (0-100).

        Returns:
            The p-th percentile value.
        """
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[-1]
        d = k - f
        return sorted_data[f] * (1 - d) + sorted_data[c] * d


def estimate_memory_usage(obj: Any) -> int:
    """Estimate memory usage of a Python object recursively.

    This is an approximation using sys.getsizeof with basic recursion
    for common container types.

    Args:
        obj: The object to measure.

    Returns:
        Estimated memory usage in bytes.
    """
    seen = set()
    total = 0

    def _sizeof(o: Any) -> int:
        nonlocal total
        obj_id = id(o)
        if obj_id in seen:
            return 0
        seen.add(obj_id)
        size = sys.getsizeof(o)
        total += size

        if isinstance(o, dict):
            for k, v in o.items():
                _sizeof(k)
                _sizeof(v)
        elif isinstance(o, (list, tuple, set, frozenset)):
            for item in o:
                _sizeof(item)

        return size

    _sizeof(obj)
    return total


class ResourceSampler:
    """Background thread that samples CPU% and RSS at a fixed interval.

    Feeds samples into a MetricsCollector via record_resource_sample().

    Usage:
        sampler = ResourceSampler(collector, interval=0.1)
        sampler.start()
        # ... run benchmark ...
        sampler.stop()
    """

    def __init__(self, collector: "MetricsCollector", interval: float = 0.1) -> None:
        self.collector = collector
        self.interval = interval
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if _HAS_PSUTIL:
            self._proc = psutil.Process(os.getpid())
            # Prime cpu_percent (first call returns 0.0)
            self._proc.cpu_percent(interval=None)
        else:
            self._proc = None

    def _run(self) -> None:
        while not self._stop.is_set():
            if self._proc is not None:
                try:
                    cpu = self._proc.cpu_percent(interval=None)
                    rss_mb = self._proc.memory_info().rss / (1024 * 1024)
                    self.collector.record_resource_sample(cpu, rss_mb)
                except Exception:
                    pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        if self._proc is None:
            return  # psutil unavailable; silently no-op
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
