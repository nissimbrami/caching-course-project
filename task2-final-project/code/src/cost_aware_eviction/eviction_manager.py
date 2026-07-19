"""GDSF (Greedy Dual-Size Frequency) Eviction Manager.

Implements cost-aware cache eviction using the priority formula:
    Priority(i) = Clock + (freq(i)^alpha * cost(i)^beta) / size(i)

The clock advances to the priority of the last evicted item, ensuring that
newly inserted items always have higher priority than the clock value.
"""

import threading
from typing import Any, Dict, List, Optional

from .config import GDSFConfig
from .priority_queue import IndexedMinHeap


class GDSFEvictionManager:
    """Cost-aware eviction manager using the GDSF algorithm.

    This manager maintains a priority queue of cached items, where priority
    is computed based on access frequency, retrieval cost, and item size.
    Items with the lowest priority are evicted first when the cache is full.

    Thread-safety is guaranteed via a threading.Lock on all public methods.
    """

    def __init__(
        self,
        max_size: int = 1000,
        alpha: float = 1.0,
        beta: float = 1.0,
        config: Optional[GDSFConfig] = None,
    ) -> None:
        """Initialize the GDSF eviction manager.

        Args:
            max_size: Maximum cache size in bytes.
            alpha: Frequency weight exponent (default 1.0).
            beta: Cost weight exponent (default 1.0).
            config: Optional full configuration. If provided, max_size, alpha,
                and beta parameters are ignored in favor of config values.
        """
        if config is not None:
            self._config = config
        else:
            self._config = GDSFConfig(max_size=max_size, alpha=alpha, beta=beta)

        self._heap = IndexedMinHeap()
        self._metadata: Dict[Any, Dict] = {}  # key -> {freq, cost, size}
        self._current_size: int = 0
        self._clock: float = 0.0
        self._lock = threading.Lock()

    @property
    def clock(self) -> float:
        """Return the current clock value."""
        return self._clock

    @property
    def current_size(self) -> int:
        """Return the current total size of cached items in bytes."""
        return self._current_size

    @property
    def max_size(self) -> int:
        """Return the maximum cache capacity."""
        return self._config.max_size

    @property
    def utilization(self) -> float:
        """Current cache utilization as a fraction (0.0 to 1.0)."""
        if self._config.max_size <= 0:
            return 0.0
        return self._current_size / self._config.max_size

    def __len__(self) -> int:
        """Return the number of items in the cache."""
        return len(self._heap)

    def __contains__(self, key: Any) -> bool:
        """Check if a key is in the cache."""
        return key in self._metadata

    def put(self, key: Any, size: Optional[int] = None, cost: Optional[float] = None) -> List[Any]:
        """Insert an item into the cache, evicting items if necessary.

        If the key already exists, updates the metadata (cost, size) and
        increments the frequency counter without duplicating.

        Args:
            key: The cache key to insert.
            size: The size of the item in bytes (default: config.default_size).
            cost: The retrieval cost of the item (default: config.default_cost).

        Returns:
            A list of keys that were evicted to make room.
        """
        with self._lock:
            if size is None:
                size = self._config.default_size
            if cost is None:
                cost = self._config.default_cost

            if size <= 0:
                raise ValueError(
                    f"size must be a positive integer, got {size}"
                )
            if cost < 0:
                raise ValueError(
                    f"cost must be non-negative, got {cost}"
                )

            # Handle duplicate key - update in place
            if key in self._metadata:
                meta = self._metadata[key]
                old_size = meta["size"]
                meta["freq"] += 1
                meta["cost"] = cost
                meta["size"] = size
                self._current_size += (size - old_size)
                # Recompute priority
                new_priority = self._compute_priority(
                    meta["freq"], meta["cost"], meta["size"]
                )
                self._heap.update(key, new_priority)
                return []

            # If item is larger than total cache capacity, reject it
            if size > self._config.max_size:
                return []

            # Evict until there's enough space
            evicted = []
            while self._current_size + size > self._config.max_size:
                evicted_key = self._evict_one()
                evicted.append(evicted_key)

            # Insert the new item
            freq = 1
            priority = self._compute_priority(freq, cost, size)
            self._heap.push(key, priority)
            self._metadata[key] = {"freq": freq, "cost": cost, "size": size}
            self._current_size += size

            return evicted

    def access(self, key: Any) -> bool:
        """Record an access to an existing cached item.

        Increments the frequency counter and recomputes the priority.

        Args:
            key: The cache key that was accessed.

        Returns:
            True if the key exists and was updated, False otherwise.
        """
        with self._lock:
            if key not in self._metadata:
                return False

            meta = self._metadata[key]
            meta["freq"] += 1
            new_priority = self._compute_priority(
                meta["freq"], meta["cost"], meta["size"]
            )
            self._heap.update(key, new_priority)
            return True

    def remove(self, key: Any) -> bool:
        """Explicitly remove an item from the cache.

        Args:
            key: The cache key to remove.

        Returns:
            True if the key existed and was removed, False otherwise.
        """
        with self._lock:
            if key not in self._metadata:
                return False

            self._heap.remove(key)
            meta = self._metadata.pop(key)
            self._current_size -= meta["size"]
            return True

    def get_priority(self, key: Any) -> Optional[float]:
        """Get the current priority of a cached item.

        Args:
            key: The cache key.

        Returns:
            The priority value, or None if the key is not in the cache.
        """
        with self._lock:
            if key not in self._metadata:
                return None
            meta = self._metadata[key]
            return self._compute_priority(meta["freq"], meta["cost"], meta["size"])

    def _compute_priority(self, freq: int, cost: float, size: int) -> float:
        """Compute GDSF priority for an item.

        Priority(i) = Clock + (freq(i)^alpha * cost(i)^beta) / size(i)

        Args:
            freq: Access frequency of the item.
            cost: Retrieval cost of the item.
            size: Size of the item.

        Returns:
            The computed priority value (higher = more important to keep).
        """
        numerator = (freq ** self._config.alpha) * (cost ** self._config.beta)
        denominator = max(size, 1)  # avoid division by zero
        return self._clock + numerator / denominator

    def evict_one(self) -> Optional[Any]:
        """Evict the item with the lowest priority (thread-safe).

        Acquires the internal lock, checks if the heap is empty, and if not,
        evicts the lowest-priority item. The clock is advanced to the priority
        of the evicted item (GDSF inflation).

        Returns:
            The key of the evicted item, or None if the cache is empty.
        """
        with self._lock:
            if len(self._heap) == 0:
                return None
            return self._evict_one()

    def _evict_one(self) -> Any:
        """Evict the item with the lowest priority.

        Advances the clock to the priority of the evicted item (GDSF inflation).

        Returns:
            The key of the evicted item.
        """
        key, priority = self._heap.pop()
        # Advance clock to evicted item's priority
        self._clock = priority
        meta = self._metadata.pop(key)
        self._current_size -= meta["size"]
        return key
