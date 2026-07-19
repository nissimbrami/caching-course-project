"""Property-based tests for the GDSF eviction manager using Hypothesis.

These tests establish invariants that must hold across a wide range of
randomly-generated inputs. They complement the example-based tests in
test_eviction_manager.py by exercising edge cases the developer may not
have thought of.

Invariants tested:
1. Capacity: total tracked size never exceeds max_size after any operation.
2. Monotone clock: the clock value is non-decreasing across evictions.
3. Uniqueness: putting the same key twice leaves cache with a single entry.
4. Determinism: identical seeds + identical operations yield identical state.
5. Priority ordering: evict_one() always removes the lowest-priority key.
6. Empty semantics: evict_one() on empty cache returns None cleanly.
"""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck

from src.cost_aware_eviction.eviction_manager import GDSFEvictionManager


# ----- Strategies ---------------------------------------------------------

keys = st.integers(min_value=0, max_value=200).map(lambda i: f"k{i}")
sizes = st.integers(min_value=1, max_value=50)
costs = st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
alphas = st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False)
betas = st.floats(min_value=0.1, max_value=3.0, allow_nan=False, allow_infinity=False)

put_op = st.tuples(st.just("put"), keys, sizes, costs)
get_op = st.tuples(st.just("get"), keys, st.just(0), st.just(0.0))
op = st.one_of(put_op, get_op)


# ----- Property tests -----------------------------------------------------

@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ops=st.lists(op, min_size=1, max_size=80),
    max_size=st.integers(min_value=200, max_value=2000),
    alpha=alphas,
    beta=betas,
)
def test_capacity_invariant_never_violated(ops, max_size, alpha, beta):
    """current_size must never exceed max_size, no matter the operation sequence.

    Note: We bound the range so single-item sizes (max 50) cannot exceed max_size
    on their own; the interesting case is that fill-then-evict logic keeps totals
    bounded across long sequences.
    """
    mgr = GDSFEvictionManager(max_size=max_size, alpha=alpha, beta=beta)

    for kind, key, size, cost in ops:
        if kind == "put":
            mgr.put(key, size=size, cost=cost)
        else:
            mgr.access(key)
        assert mgr.current_size <= max_size, (
            f"Capacity violated: current_size={mgr.current_size} > max_size={max_size}"
        )


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ops=st.lists(put_op, min_size=1, max_size=60),
    max_size=st.integers(min_value=10, max_value=200),
)
def test_clock_is_monotone_non_decreasing(ops, max_size):
    """The eviction clock must never decrease."""
    mgr = GDSFEvictionManager(max_size=max_size, alpha=1.0, beta=1.0)
    last_clock = mgr.clock
    for _, key, size, cost in ops:
        mgr.put(key, size=size, cost=cost)
        assert mgr.clock >= last_clock, f"Clock went backwards: {mgr.clock} < {last_clock}"
        last_clock = mgr.clock


@settings(max_examples=100, deadline=None)
@given(key=keys, size=sizes, cost=costs, repeats=st.integers(min_value=2, max_value=10))
def test_duplicate_puts_do_not_multiply(key, size, cost, repeats):
    """Putting the same key N times leaves at most one entry in the cache."""
    mgr = GDSFEvictionManager(max_size=10_000, alpha=1.0, beta=1.0)
    for _ in range(repeats):
        mgr.put(key, size=size, cost=cost)
    assert len(mgr) == 1
    assert key in mgr
    assert mgr.current_size == size


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ops=st.lists(put_op, min_size=5, max_size=40),
    max_size=st.integers(min_value=10, max_value=100),
)
def test_determinism_same_ops_same_state(ops, max_size):
    """Running the same operations in two managers yields the same key set and size."""
    mgr1 = GDSFEvictionManager(max_size=max_size, alpha=1.0, beta=1.0)
    mgr2 = GDSFEvictionManager(max_size=max_size, alpha=1.0, beta=1.0)

    for _, key, size, cost in ops:
        mgr1.put(key, size=size, cost=cost)
        mgr2.put(key, size=size, cost=cost)

    keys1 = {k for k in mgr1._metadata}  # noqa: SLF001
    keys2 = {k for k in mgr2._metadata}  # noqa: SLF001
    assert keys1 == keys2
    assert mgr1.current_size == mgr2.current_size
    assert mgr1.clock == mgr2.clock


def test_evict_one_on_empty_returns_none():
    """evict_one on an empty cache is a no-op returning None."""
    mgr = GDSFEvictionManager(max_size=100)
    assert mgr.evict_one() is None


@settings(max_examples=50, deadline=None)
@given(
    entries=st.lists(
        st.tuples(keys, sizes, costs),
        min_size=2,
        max_size=20,
        unique_by=lambda t: t[0],
    ),
)
def test_evict_one_returns_lowest_priority_key(entries):
    """evict_one must remove the key with the smallest priority value."""
    total_size = sum(s for _, s, _ in entries)
    # Give enough capacity so no auto-eviction fires while filling.
    mgr = GDSFEvictionManager(max_size=total_size + 1, alpha=1.0, beta=1.0)
    for key, size, cost in entries:
        mgr.put(key, size=size, cost=cost)

    # Snapshot priorities before evict.
    priorities = {k: mgr.get_priority(k) for k, _, _ in entries}
    expected_victim = min(priorities, key=priorities.get)

    victim = mgr.evict_one()
    assert victim == expected_victim, (
        f"Expected to evict lowest-priority key {expected_victim} "
        f"(prio={priorities[expected_victim]}), got {victim} "
        f"(prio={priorities.get(victim)})"
    )
