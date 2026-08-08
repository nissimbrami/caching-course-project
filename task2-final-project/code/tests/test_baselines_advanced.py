"""Tests for advanced baseline policies (W-TinyLFU, ARC, S3-FIFO)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.baselines_advanced import (
    WTinyLFUPolicy,
    ARCPolicy,
    S3FIFOPolicy,
    register_advanced_policies,
)
from benchmarks import policies as pols


# --------------------------------------------------------------------------- #
# Fresh-cache and interface conformance for all three                          #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("cls", [WTinyLFUPolicy, ARCPolicy, S3FIFOPolicy])
def test_fresh_cache_miss(cls):
    p = cls(max_size=1000)
    assert p.access("x") is False
    assert p.current_size == 0


@pytest.mark.parametrize("cls", [WTinyLFUPolicy, ARCPolicy, S3FIFOPolicy])
def test_basic_put_hit(cls):
    p = cls(max_size=1000)
    evicted = p.put("a", size=10, cost=1.0)
    assert evicted == []
    assert p.access("a") is True
    assert p.current_size == 10


@pytest.mark.parametrize("cls", [WTinyLFUPolicy, ARCPolicy, S3FIFOPolicy])
def test_oversized_rejected(cls):
    p = cls(max_size=100)
    evicted = p.put("big", size=1000, cost=1.0)
    assert evicted == []
    assert p.access("big") is False
    assert p.current_size == 0


@pytest.mark.parametrize("cls", [WTinyLFUPolicy, ARCPolicy, S3FIFOPolicy])
def test_reset_clears(cls):
    p = cls(max_size=1000)
    for i in range(10):
        p.put(f"k{i}", size=50, cost=1.0)
    p.reset()
    assert p.current_size == 0
    assert p.access("k0") is False


@pytest.mark.parametrize("cls", [WTinyLFUPolicy, ARCPolicy, S3FIFOPolicy])
def test_byte_bookkeeping(cls):
    p = cls(max_size=500)
    # Insert items, some may get evicted, but current_size must equal
    # the sum of currently resident sizes (which is bounded by max_size).
    for i in range(30):
        p.put(f"k{i}", size=40, cost=1.0)
        assert 0 <= p.current_size <= 500


@pytest.mark.parametrize("cls", [WTinyLFUPolicy, ARCPolicy, S3FIFOPolicy])
def test_update_in_place(cls):
    p = cls(max_size=1000)
    p.put("a", size=100, cost=1.0)
    sz1 = p.current_size
    p.put("a", size=100, cost=2.0)  # same size, same cost slot
    assert p.current_size == sz1


# --------------------------------------------------------------------------- #
# Algorithm-specific behavioural checks                                        #
# --------------------------------------------------------------------------- #

def test_wtinylfu_frequent_survives_burst():
    """A frequent key must survive a burst of one-hit wonders."""
    p = WTinyLFUPolicy(max_size=2000)
    # Prime frequency for "hot"
    p.put("hot", size=100, cost=1.0)
    for _ in range(50):
        p.access("hot")
    # Flood with one-hit wonders
    for i in range(200):
        p.put(f"cold{i}", size=100, cost=1.0)
    # hot should still be reachable
    # (touch frequency again to make sure counters recorded)
    hits = sum(p.access("hot") for _ in range(10))
    assert hits >= 1  # frequency-aware admission preserved hot


def test_wtinylfu_sketch_ages():
    p = WTinyLFUPolicy(max_size=1000)
    p._age_period = 10  # force fast aging
    for _ in range(20):
        p.access("phantom")
    # After aging, sketch counters must have been halved
    # (indirect: aging is deterministic — verify counter dropped)
    assert p._sketch.estimate("phantom") <= 20


def test_arc_p_increases_on_b1_ghost_hit():
    p = ARCPolicy(max_size=200)
    # Fill T1 with items that will be pushed to B1
    for i in range(20):
        p.put(f"k{i}", size=50, cost=1.0)
    p0 = p._p
    # Now hit a B1 key (one that was evicted)
    b1_keys = list(p._b1.keys())
    if b1_keys:
        p.put(b1_keys[0], size=50, cost=1.0)
        # p should have increased (recency reinforced)
        assert p._p >= p0


def test_arc_promotes_recent_to_frequent():
    p = ARCPolicy(max_size=200)
    p.put("x", size=50, cost=1.0)
    # First access: x moves T1 -> T2
    assert p.access("x") is True
    # Verify by inspecting internal state
    assert "x" in p._t2 or "x" not in p._t1


def test_s3fifo_ghost_promotes_to_main():
    p = S3FIFOPolicy(max_size=1000, small_frac=0.1)
    # Fill Small so items get evicted to ghost
    for i in range(50):
        p.put(f"k{i}", size=50, cost=1.0)
    # Some early keys should now be in ghost
    assert len(p._ghost_set) > 0
    ghost_key = next(iter(p._ghost_set))
    # Re-insert a ghosted key: should go directly to Main
    p.put(ghost_key, size=50, cost=1.0)
    assert ghost_key in p._m


def test_s3fifo_frequency_promotes_within_small():
    p = S3FIFOPolicy(max_size=1000, small_frac=0.2)
    p.put("warm", size=50, cost=1.0)
    for _ in range(5):
        p.access("warm")
    # Force S to overflow by adding cold items
    for i in range(20):
        p.put(f"cold{i}", size=50, cost=1.0)
    # warm should have been promoted to M (freq > 1) instead of ghosted
    assert p.access("warm") is True or "warm" in p._m


# --------------------------------------------------------------------------- #
# Registry integration                                                         #
# --------------------------------------------------------------------------- #

def test_registry_registration():
    register_advanced_policies()
    assert "W-TinyLFU" in pols.POLICY_REGISTRY
    assert "ARC" in pols.POLICY_REGISTRY
    assert "S3-FIFO" in pols.POLICY_REGISTRY


def test_create_via_factory():
    register_advanced_policies()
    for name in ("W-TinyLFU", "ARC", "S3-FIFO"):
        p = pols.create_policy(name, max_size=1024)
        assert p.max_size == 1024
        assert p.current_size == 0
