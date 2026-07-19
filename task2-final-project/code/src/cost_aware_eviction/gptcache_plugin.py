"""GPTCache integration plugin for GDSF eviction.

Provides an adapter that bridges GPTCache's eviction interface to the GDSF
eviction manager. This allows GPTCache to use cost-aware eviction decisions
based on Greedy Dual-Size Frequency.

GPTCache's EvictionBase interface expects:
    - put(objs: List[Any]) -> None
    - get() -> Any (returns the key to evict)
    - is_evict() -> bool

This plugin adapts that interface to work with our GDSF manager, which needs
additional metadata (size, cost) for each entry.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from .eviction_manager import GDSFEvictionManager
from .cost_estimator import CostEstimator
from .config import GDSFConfig

logger = logging.getLogger(__name__)


# Try to import GPTCache's base class
try:
    from gptcache.manager.eviction.base import EvictionBase

    _GPTCACHE_AVAILABLE = True
except ImportError:
    _GPTCACHE_AVAILABLE = False

    # Define a compatible base class when GPTCache is not installed
    class EvictionBase:  # type: ignore[no-redef]
        """Fallback base class when GPTCache is not installed."""

        def put(self, objs: List[Any]) -> None:
            raise NotImplementedError

        def get(self) -> Any:
            raise NotImplementedError

        def is_evict(self) -> bool:
            raise NotImplementedError


class GDSFEvictionPlugin(EvictionBase):
    """GPTCache-compatible eviction plugin using GDSF strategy.

    This adapter bridges GPTCache's simple eviction interface with the GDSF
    manager's cost-aware eviction logic. Since GPTCache's interface does not
    natively support passing size and cost metadata, this plugin provides
    multiple strategies for obtaining that information:

    1. Metadata callback: A user-provided function that returns (size, cost)
       for a given key.
    2. Default estimation: Uses the CostEstimator with default values.
    3. Manual registration: Users can pre-register metadata via
       `register_metadata()` before items enter the cache.

    Example usage with GPTCache:
        ```python
        from gptcache import Cache
        from src.cost_aware_eviction import GDSFEvictionPlugin

        plugin = GDSFEvictionPlugin(max_size=1024*1024*50)  # 50MB
        cache = Cache()
        cache.init(eviction_manager=plugin)
        ```

    Example usage standalone:
        ```python
        plugin = GDSFEvictionPlugin(max_size=10000)
        plugin.register_metadata("key1", size=500, cost=0.01)
        plugin.put(["key1"])
        plugin.register_metadata("key2", size=300, cost=0.005)
        plugin.put(["key2"])

        if plugin.is_evict():
            evicted_key = plugin.get()
        ```
    """

    def __init__(
        self,
        max_size: int = 1024 * 1024 * 100,
        alpha: float = 1.0,
        beta: float = 1.0,
        config: Optional[GDSFConfig] = None,
        metadata_callback: Optional[Callable[[Any], tuple]] = None,
        default_entry_size: int = 1024,
        default_entry_cost: float = 0.001,
        **kwargs: Any,
    ) -> None:
        """Initialize the GDSF eviction plugin.

        Args:
            max_size: Maximum cache size in bytes.
            alpha: Frequency weight exponent.
            beta: Cost weight exponent.
            config: Optional full GDSF configuration.
            metadata_callback: Optional callable that takes a key and returns
                a tuple of (size: int, cost: float). Called when metadata is
                not pre-registered.
            default_entry_size: Default entry size in bytes when metadata is
                unavailable.
            default_entry_cost: Default entry cost when metadata is unavailable.
            **kwargs: Additional keyword arguments (for GPTCache compatibility).
        """
        self._manager = GDSFEvictionManager(
            max_size=max_size, alpha=alpha, beta=beta, config=config
        )
        self._cost_estimator = CostEstimator(config=config or GDSFConfig())
        self._metadata_callback = metadata_callback
        self._default_entry_size = default_entry_size
        self._default_entry_cost = default_entry_cost
        self._pending_metadata: Dict[Any, Dict[str, Any]] = {}
        self._eviction_queue: List[Any] = []

    def register_metadata(
        self,
        key: Any,
        size: Optional[int] = None,
        cost: Optional[float] = None,
        response_text: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        """Pre-register metadata for a cache entry before it is put.

        This method allows users to provide size and cost information for
        entries that will be added via the standard `put()` interface.

        Args:
            key: The cache key to register metadata for.
            size: Size of the entry in bytes. If None and response_text is
                provided, size is estimated from the text length.
            cost: Regeneration cost. If None and response_text is provided,
                cost is estimated using the CostEstimator.
            response_text: The cached response text (used for estimation).
            model_name: The model that generated the response.
        """
        entry_size = size
        entry_cost = cost

        if entry_size is None:
            if response_text is not None:
                entry_size = len(response_text.encode("utf-8"))
            else:
                entry_size = self._default_entry_size

        if entry_cost is None:
            if response_text is not None:
                entry_cost = self._cost_estimator.estimate_cost(
                    response_text, model_name
                )
            else:
                entry_cost = self._default_entry_cost

        self._pending_metadata[key] = {"size": entry_size, "cost": entry_cost}

    def _get_metadata(self, key: Any) -> tuple:
        """Get size and cost metadata for a key.

        Checks pending metadata first, then tries the callback, then uses
        defaults.

        Args:
            key: The cache key.

        Returns:
            Tuple of (size, cost).
        """
        # Check pre-registered metadata
        if key in self._pending_metadata:
            meta = self._pending_metadata.pop(key)
            return (meta["size"], meta["cost"])

        # Try the metadata callback
        if self._metadata_callback is not None:
            try:
                result = self._metadata_callback(key)
                if isinstance(result, tuple) and len(result) == 2:
                    return result
            except Exception as e:
                logger.warning(f"Metadata callback failed for key '{key}': {e}")

        # Fall back to defaults
        return (self._default_entry_size, self._default_entry_cost)

    def put(self, objs: List[Any]) -> None:
        """Add entries to the eviction manager.

        This is the standard GPTCache interface for notifying the eviction
        manager about new cache entries. If size/cost metadata has been
        pre-registered (via `register_metadata()`), it will be used.
        Otherwise, defaults are applied.

        Args:
            objs: List of cache keys being added.
        """
        for key in objs:
            size, cost = self._get_metadata(key)
            evicted = self._manager.put(key, size, cost)
            # Store evicted keys for retrieval via get()
            self._eviction_queue.extend(evicted)

    def get(self) -> Any:
        """Get the next key to evict.

        In the GDSF model, evictions happen proactively during put(). This
        method returns keys that have already been evicted during the last
        put() call(s). If no evictions are pending, it evicts the lowest
        priority item.

        Returns:
            The key of the evicted entry, or None if cache is empty.
        """
        if self._eviction_queue:
            return self._eviction_queue.pop(0)

        # If called independently (not after a put), evict the lowest priority
        return self._manager.evict_one()

    def is_evict(self) -> bool:
        """Check whether eviction is needed.

        Returns True if the cache is at capacity or if there are pending
        evictions from a previous put() call.

        Returns:
            True if eviction should occur, False otherwise.
        """
        if self._eviction_queue:
            return True
        return self._manager.utilization >= 1.0

    def access(self, key: Any) -> bool:
        """Notify the plugin that a cache hit occurred.

        Updates the frequency and priority of the accessed entry.

        Args:
            key: The key that was accessed (cache hit).

        Returns:
            True if the key was found and updated, False otherwise.
        """
        return self._manager.access(key)

    @property
    def current_size(self) -> int:
        """Current total size of cached entries in bytes."""
        return self._manager.current_size

    @property
    def utilization(self) -> float:
        """Current cache utilization as a fraction (0.0 to 1.0)."""
        return self._manager.utilization

    @property
    def num_entries(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._manager)

    def __len__(self) -> int:
        """Return the number of entries in the cache."""
        return len(self._manager)
