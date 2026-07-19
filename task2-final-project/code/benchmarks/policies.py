"""
Cache eviction policy implementations for benchmarking.

Provides a common interface (CachePolicy) and implementations of standard
eviction policies (LRU, FIFO, LFU, Random) plus the GDSF cost-aware policy.
All policies track current used size and support eviction based on a maximum
byte capacity.
"""

from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import heapq
import random
import math


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


@dataclass(order=True)
class _GDSFEntry:
    """Internal entry for the GDSF priority queue."""
    priority: float
    key: str = field(compare=False)
    size: int = field(compare=False)
    cost: float = field(compare=False)
    freq: int = field(compare=False)
    valid: bool = field(default=True, compare=False)


class GDSFPolicy(CachePolicy):
    """Greedy Dual-Size Frequency (GDSF) eviction policy.

    Priority = Clock + freq^alpha * cost^beta / size

    This is the cost-aware eviction policy that forms the core enhancement
    of this project. It generalizes LRU/LFU by incorporating both the cost
    of regenerating an item and its size into eviction decisions.

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
        self._clock: float = 0.0  # Aging mechanism (inflation counter)
        self._heap: List[_GDSFEntry] = []  # Min-heap by priority
        self._entries: Dict[str, _GDSFEntry] = {}  # key -> current entry
        self._freq: Dict[str, int] = defaultdict(int)  # key -> access count

    def _compute_priority(self, freq: int, cost: float, size: int) -> float:
        """Compute GDSF priority value.

        Priority = Clock + freq^alpha * cost^beta / size
        Higher priority = more valuable = evicted later.
        """
        size_factor = max(size, 1)  # avoid division by zero
        return self._clock + (freq ** self._alpha) * (cost ** self._beta) / size_factor

    def put(self, key: str, size: int, cost: float) -> List[str]:
        evicted: List[str] = []

        # Remove existing entry if present
        if key in self._entries:
            old_entry = self._entries[key]
            old_entry.valid = False  # Mark as invalid (lazy deletion)
            self.current_size -= old_entry.size
            del self._entries[key]

        # Evict until there is room
        while self.current_size + size > self.max_size and self._entries:
            # Pop the minimum priority entry
            while self._heap:
                entry = heapq.heappop(self._heap)
                if entry.valid:
                    # This is the victim
                    self._clock = entry.priority  # Advance clock
                    self.current_size -= entry.size
                    del self._entries[entry.key]
                    if entry.key in self._freq:
                        del self._freq[entry.key]
                    evicted.append(entry.key)
                    break
            else:
                # Heap exhausted (shouldn't happen if entries exist)
                break

        # Insert new entry
        if size <= self.max_size:
            self._freq[key] = 1
            priority = self._compute_priority(1, cost, size)
            entry = _GDSFEntry(
                priority=priority,
                key=key,
                size=size,
                cost=cost,
                freq=1,
            )
            heapq.heappush(self._heap, entry)
            self._entries[key] = entry
            self.current_size += size

        return evicted

    def access(self, key: str) -> bool:
        if key not in self._entries:
            return False

        # Update frequency and recompute priority
        old_entry = self._entries[key]
        old_entry.valid = False  # Invalidate old heap entry

        self._freq[key] += 1
        new_priority = self._compute_priority(
            self._freq[key], old_entry.cost, old_entry.size
        )

        new_entry = _GDSFEntry(
            priority=new_priority,
            key=key,
            size=old_entry.size,
            cost=old_entry.cost,
            freq=self._freq[key],
        )
        heapq.heappush(self._heap, new_entry)
        self._entries[key] = new_entry

        return True

    def reset(self) -> None:
        self._clock = 0.0
        self._heap.clear()
        self._entries.clear()
        self._freq.clear()
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
