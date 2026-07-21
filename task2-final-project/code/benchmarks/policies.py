"""
Cache eviction policy implementations for benchmarking.

Provides a common interface (CachePolicy) and implementations of standard
eviction policies (LRU, FIFO, LFU, Random) plus the GDSF cost-aware policy.
All policies track current used size and support eviction based on a maximum
byte capacity.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
import random

from src.cost_aware_eviction import GDSFEvictionManager


class CachePolicy(ABC):
    """Abstract base class for cache eviction policies.

    All policies manage a cache with a fixed byte-level capacity (max_size).
    Items are inserted with a key, byte size, and associated cost. When the
    cache is full, the policy decides which items to evict.
    """

    def __init__(self, max_size: int) -> None:
        """Initialize the cache policy.

        Args:
            max_size: Maximum total size of cached items in bytes.
        """
        self.max_size = max_size
        self.current_size = 0

    @abstractmethod
    def put(self, key: str, size: int, cost: float) -> List[str]:
        """Insert an item into the cache, evicting as needed.

        If the item already exists, it is treated as an update (size/cost
        may change). Returns the list of keys that were evicted to make room.

        Args:
            key: Unique identifier for the cache entry.
            size: Size of the entry in bytes.
            cost: Cost of regenerating this entry (e.g., API dollar cost).

        Returns:
            List of evicted keys (may be empty if no eviction was needed).
        """
        ...

    @abstractmethod
    def access(self, key: str) -> bool:
        """Record an access (hit) for an existing cache entry.

        Args:
            key: The key being accessed.

        Returns:
            True if the key is in the cache (hit), False otherwise (miss).
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the cache to its initial empty state."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of the policy."""
        ...

    def __repr__(self) -> str:
        return f"{self.name}(max_size={self.max_size}, current_size={self.current_size})"


class LRUPolicy(CachePolicy):
    """Least Recently Used eviction policy.

    Evicts the item that has not been accessed for the longest time.
    Uses an OrderedDict for O(1) access and eviction.
    """

    def __init__(self, max_size: int) -> None:
        super().__init__(max_size)
        self._cache: OrderedDict[str, Tuple[int, float]] = OrderedDict()  # key -> (size, cost)

    def put(self, key: str, size: int, cost: float) -> List[str]:
        evicted: List[str] = []

        # If key already exists, remove it first (will be re-inserted)
        if key in self._cache:
            old_size, _ = self._cache.pop(key)
            self.current_size -= old_size

        # Evict until there is room
        while self.current_size + size > self.max_size and self._cache:
            evicted_key, (evicted_size, _) = self._cache.popitem(last=False)
            self.current_size -= evicted_size
            evicted.append(evicted_key)

        # Insert the new item
        if size <= self.max_size:
            self._cache[key] = (size, cost)
            self.current_size += size

        return evicted

    def access(self, key: str) -> bool:
        if key in self._cache:
            self._cache.move_to_end(key)
            return True
        return False

    def reset(self) -> None:
        self._cache.clear()
        self.current_size = 0

    @property
    def name(self) -> str:
        return "LRU"


class FIFOPolicy(CachePolicy):
    """First In, First Out eviction policy.

    Evicts the item that was inserted earliest, regardless of access pattern.
    """

    def __init__(self, max_size: int) -> None:
        super().__init__(max_size)
        self._cache: OrderedDict[str, Tuple[int, float]] = OrderedDict()

    def put(self, key: str, size: int, cost: float) -> List[str]:
        evicted: List[str] = []

        if key in self._cache:
            old_size, _ = self._cache.pop(key)
            self.current_size -= old_size

        while self.current_size + size > self.max_size and self._cache:
            evicted_key, (evicted_size, _) = self._cache.popitem(last=False)
            self.current_size -= evicted_size
            evicted.append(evicted_key)

        if size <= self.max_size:
            self._cache[key] = (size, cost)
            self.current_size += size

        return evicted

    def access(self, key: str) -> bool:
        # FIFO does not reorder on access
        return key in self._cache

    def reset(self) -> None:
        self._cache.clear()
        self.current_size = 0

    @property
    def name(self) -> str:
        return "FIFO"


class LFUPolicy(CachePolicy):
    """Least Frequently Used eviction policy.

    Evicts the item with the lowest access frequency. Ties broken by
    insertion order (oldest first).
    """

    def __init__(self, max_size: int) -> None:
        super().__init__(max_size)
        self._cache: Dict[str, Tuple[int, float]] = {}  # key -> (size, cost)
        self._freq: Dict[str, int] = {}  # key -> access count
        self._insertion_order: Dict[str, int] = {}  # key -> insertion timestamp
        self._counter: int = 0

    def _find_lfu_key(self) -> Optional[str]:
        """Find the key with the lowest frequency (ties broken by oldest insertion)."""
        if not self._cache:
            return None
        min_freq = min(self._freq[k] for k in self._cache)
        candidates = [k for k in self._cache if self._freq[k] == min_freq]
        # Break ties by insertion order
        return min(candidates, key=lambda k: self._insertion_order[k])

    def put(self, key: str, size: int, cost: float) -> List[str]:
        evicted: List[str] = []

        if key in self._cache:
            old_size, _ = self._cache[key]
            self.current_size -= old_size
            del self._cache[key]

        while self.current_size + size > self.max_size and self._cache:
            victim = self._find_lfu_key()
            if victim is None:
                break
            evicted_size, _ = self._cache.pop(victim)
            del self._freq[victim]
            del self._insertion_order[victim]
            self.current_size -= evicted_size
            evicted.append(victim)

        if size <= self.max_size:
            self._cache[key] = (size, cost)
            self._freq[key] = 1
            self._counter += 1
            self._insertion_order[key] = self._counter
            self.current_size += size

        return evicted

    def access(self, key: str) -> bool:
        if key in self._cache:
            self._freq[key] += 1
            return True
        return False

    def reset(self) -> None:
        self._cache.clear()
        self._freq.clear()
        self._insertion_order.clear()
        self._counter = 0
        self.current_size = 0

    @property
    def name(self) -> str:
        return "LFU"


class RandomPolicy(CachePolicy):
    """Random eviction policy.

    Evicts a randomly chosen item when the cache is full. Useful as a
    lower-bound baseline.
    """

    def __init__(self, max_size: int, seed: int = 42) -> None:
        super().__init__(max_size)
        self._cache: Dict[str, Tuple[int, float]] = {}
        self._rng = random.Random(seed)

    def put(self, key: str, size: int, cost: float) -> List[str]:
        evicted: List[str] = []

        if key in self._cache:
            old_size, _ = self._cache.pop(key)
            self.current_size -= old_size

        while self.current_size + size > self.max_size and self._cache:
            victim = self._rng.choice(list(self._cache.keys()))
            evicted_size, _ = self._cache.pop(victim)
            self.current_size -= evicted_size
            evicted.append(victim)

        if size <= self.max_size:
            self._cache[key] = (size, cost)
            self.current_size += size

        return evicted

    def access(self, key: str) -> bool:
        return key in self._cache

    def reset(self) -> None:
        self._cache.clear()
        self.current_size = 0

    @property
    def name(self) -> str:
        return "Random"


class GDSFPolicy(CachePolicy):
    """Greedy Dual-Size Frequency (GDSF) eviction policy.

    Priority = Clock + freq^alpha * cost^beta / size

    This is a thin adapter over the canonical GDSFEvictionManager from
    src.cost_aware_eviction so that the benchmark harness and the unit
    tests measure the exact same implementation.

    Args:
        max_size: Maximum cache capacity in bytes.
        alpha: Exponent for frequency term (default 1.0).
        beta: Exponent for cost term (default 1.0).
    """

    def __init__(
        self,
        max_size: int,
        alpha: float = 1.0,
        beta: float = 1.0,
    ) -> None:
        super().__init__(max_size)
        self._alpha = alpha
        self._beta = beta
        self._manager = GDSFEvictionManager(
            max_size=max_size, alpha=alpha, beta=beta
        )

    def put(self, key: str, size: int, cost: float) -> List[str]:
        evicted = self._manager.put(key, size=size, cost=cost)
        self.current_size = self._manager.current_size
        return list(evicted)

    def access(self, key: str) -> bool:
        hit = self._manager.access(key)
        self.current_size = self._manager.current_size
        return hit

    def reset(self) -> None:
        self._manager = GDSFEvictionManager(
            max_size=self.max_size, alpha=self._alpha, beta=self._beta
        )
        self.current_size = 0

    @property
    def name(self) -> str:
        return f"GDSF(a={self._alpha},b={self._beta})"


# Registry of all policies for easy iteration
POLICY_REGISTRY: Dict[str, type] = {
    "LRU": LRUPolicy,
    "FIFO": FIFOPolicy,
    "LFU": LFUPolicy,
    "Random": RandomPolicy,
    "GDSF": GDSFPolicy,
}


def create_policy(name: str, max_size: int, **kwargs) -> CachePolicy:
    """Create a cache policy instance by name.

    Args:
        name: Policy name (key in POLICY_REGISTRY).
        max_size: Maximum cache capacity in bytes.
        **kwargs: Additional keyword arguments for the policy constructor.

    Returns:
        Initialized CachePolicy instance.

    Raises:
        ValueError: If the policy name is not found in the registry.
    """
    if name not in POLICY_REGISTRY:
        raise ValueError(
            f"Unknown policy '{name}'. Available: {list(POLICY_REGISTRY.keys())}"
        )
    return POLICY_REGISTRY[name](max_size=max_size, **kwargs)
