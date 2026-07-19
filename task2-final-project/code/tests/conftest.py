"""Shared fixtures for GDSF test suite."""

import sys
from pathlib import Path

import pytest

# Ensure both the project root and src directory are on the Python path.
# The source module uses "from src.cost_aware_eviction.config import ..." style imports
# internally, so the project root (parent of src/) must be on sys.path.
# Tests import as "from cost_aware_eviction.X import ..." so src/ must also be on sys.path.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _PROJECT_ROOT / "src"

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from cost_aware_eviction.config import GDSFConfig
from cost_aware_eviction.priority_queue import IndexedMinHeap
from cost_aware_eviction.eviction_manager import GDSFEvictionManager
from cost_aware_eviction.cost_estimator import CostEstimator


# ---------------------------------------------------------------------------
# Configuration Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config():
    """Return a GDSFConfig with default parameters."""
    return GDSFConfig()


@pytest.fixture
def small_cache_config():
    """Return a GDSFConfig with a small cache (max_size=10)."""
    return GDSFConfig(max_size=10, alpha=1.0, beta=1.0)


@pytest.fixture
def medium_cache_config():
    """Return a GDSFConfig with a medium cache (max_size=100)."""
    return GDSFConfig(max_size=100, alpha=1.0, beta=1.0)


@pytest.fixture
def large_cache_config():
    """Return a GDSFConfig with a large cache (max_size=1000)."""
    return GDSFConfig(max_size=1000, alpha=1.0, beta=1.0)


@pytest.fixture
def alpha_zero_config():
    """Config with alpha=0 (frequency ignored)."""
    return GDSFConfig(max_size=10, alpha=0.0, beta=1.0)


@pytest.fixture
def beta_zero_config():
    """Config with beta=0 (cost ignored)."""
    return GDSFConfig(max_size=10, alpha=1.0, beta=0.0)


# ---------------------------------------------------------------------------
# Priority Queue Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def empty_heap():
    """Return an empty IndexedMinHeap."""
    return IndexedMinHeap()


@pytest.fixture
def populated_heap():
    """Return a heap with 5 elements: a(1.0), b(2.0), c(3.0), d(4.0), e(5.0)."""
    heap = IndexedMinHeap()
    heap.push("a", 1.0)
    heap.push("b", 2.0)
    heap.push("c", 3.0)
    heap.push("d", 4.0)
    heap.push("e", 5.0)
    return heap


# ---------------------------------------------------------------------------
# Eviction Manager Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_manager(small_cache_config):
    """Return a GDSFEvictionManager with max_size=10."""
    return GDSFEvictionManager(config=small_cache_config)


@pytest.fixture
def medium_manager(medium_cache_config):
    """Return a GDSFEvictionManager with max_size=100."""
    return GDSFEvictionManager(config=medium_cache_config)


@pytest.fixture
def manager_alpha_zero(alpha_zero_config):
    """Return a manager where frequency is ignored (alpha=0)."""
    return GDSFEvictionManager(config=alpha_zero_config)


@pytest.fixture
def manager_beta_zero(beta_zero_config):
    """Return a manager where cost is ignored (beta=0)."""
    return GDSFEvictionManager(config=beta_zero_config)


# ---------------------------------------------------------------------------
# Cost Estimator Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cost_estimator():
    """Return a CostEstimator with default configuration."""
    return CostEstimator()


@pytest.fixture
def cost_estimator_custom():
    """Return a CostEstimator with custom pricing."""
    config = GDSFConfig(
        model_pricing={
            "gpt-4": 0.00006,
            "gpt-3.5-turbo": 0.000002,
            "cheap-model": 0.0000001,
        }
    )
    return CostEstimator(config=config)
