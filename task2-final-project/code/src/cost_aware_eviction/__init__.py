"""Cost-Aware Eviction module implementing Greedy Dual-Size Frequency (GDSF).

This module provides a GDSF-based eviction strategy optimized for LLM response
caches. It considers frequency of access, cost of regeneration, and size of
cached entries when making eviction decisions.

Priority Formula: Priority(i) = Clock + (freq(i)^alpha * cost(i)^beta) / size(i)
"""

from .config import GDSFConfig
from .priority_queue import IndexedMinHeap
from .eviction_manager import GDSFEvictionManager
from .cost_estimator import CostEstimator
from .gptcache_plugin import GDSFEvictionPlugin

__all__ = [
    "GDSFConfig",
    "IndexedMinHeap",
    "GDSFEvictionManager",
    "CostEstimator",
    "GDSFEvictionPlugin",
]
