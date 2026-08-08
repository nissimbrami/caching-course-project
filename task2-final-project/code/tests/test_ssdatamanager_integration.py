"""Integration tests using the real gptcache.SSDataManager.

These tests prove the GDSF plugin honors GPTCache's actual eviction contract:
- get(obj) records a hit (bumps freq), doesn't select a victim
- put(objs) may trigger on_evict callback with real victims
- metadata_callback lets cost/size flow from GPTCache metadata into GDSF

Real backends are used (SQLite in-memory + FAISS in-memory) so the tests
exercise the same code paths as the full pipeline instead of mocks.
"""
from __future__ import annotations

import uuid

import numpy as np
import pytest

# Skip the whole module if the real backends are not importable.
pytest.importorskip("sqlalchemy")
pytest.importorskip("faiss")

from gptcache.manager import CacheBase, VectorBase
from gptcache.manager.data_manager import SSDataManager

from src.cost_aware_eviction.gptcache_plugin import GDSFEvictionPlugin


def _make_stack(maxsize, metadata_cb=None, on_evict=None, default_entry_size=1):
    """Build a real (SQLite + FAISS) SSDataManager wired to a GDSF plugin.

    ``default_entry_size=1`` makes the maxsize a per-entry count rather than a
    byte budget, which keeps the tests easy to reason about.
    """
    table = f"t_{uuid.uuid4().hex[:8]}"
    sqlite = CacheBase("sqlite", sql_url="sqlite:///:memory:", table_name=table)
    faiss = VectorBase("faiss", dimension=4, index_path=":memory:")
    plugin = GDSFEvictionPlugin(
        maxsize=maxsize,
        alpha=1.0,
        beta=1.0,
        default_entry_size=default_entry_size,
        default_entry_cost=1.0,
        metadata_callback=metadata_cb,
        on_evict=on_evict,
    )
    dm = SSDataManager(s=sqlite, v=faiss, o=None, e=plugin, max_size=maxsize, clean_size=1)
    return sqlite, faiss, plugin, dm


class TestSSDataManagerIntegration:

    def test_plugin_is_wired_as_eviction_base(self):
        """SSDataManager.eviction_base must be the GDSF plugin itself."""
        _, _, plugin, dm = _make_stack(3)
        assert dm.eviction_base is plugin
        assert plugin.policy == "GDSF"

    def test_get_on_hit_bumps_frequency_no_eviction(self):
        """Contract: ``plugin.get(obj)`` on an existing key is a HIT.

        A hit must bump the entry's frequency counter and return the payload
        (or None), and must not remove an entry from the cache. ``get(obj)``
        has different semantics from ``next_victim()`` and this test guards
        that boundary.
        """
        _, _, plugin, dm = _make_stack(5)
        rng = np.random.default_rng(0)
        for i in range(3):
            dm.save(f"q{i}", f"a{i}", rng.standard_normal(4).astype("float32"))

        n_before = plugin.num_entries
        first_id = min(plugin._manager._metadata.keys())
        freq_before = plugin._manager._metadata[first_id]["freq"]

        # Simulate a cache hit on the first key.
        result = plugin.get(first_id)

        # get(obj) contract: bumps frequency, does NOT evict anything.
        assert plugin.num_entries == n_before, (
            f"get() on hit must not evict: had {n_before} entries, "
            f"now have {plugin.num_entries}"
        )
        assert first_id in plugin._manager._metadata, (
            "get() on hit must not remove the hit key itself"
        )
        assert plugin._manager._metadata[first_id]["freq"] > freq_before, (
            f"get() on hit must bump freq (was {freq_before}, "
            f"now {plugin._manager._metadata[first_id]['freq']})"
        )
        # The return value is either None or the payload; it must NOT be a
        # different key (which would mean the plugin picked a victim).
        assert result != min(
            k for k in plugin._manager._metadata.keys() if k != first_id
        ), "get(obj) must not return a foreign eviction key"

    def test_overflow_triggers_on_evict_callback(self):
        """When put() causes an overflow, on_evict must fire with victim ids.

        The ``on_evict`` hook is the only channel by which SSDataManager
        learns which underlying rows to purge.
        """
        evicted = []
        _, _, plugin, dm = _make_stack(2, on_evict=lambda keys: evicted.extend(keys))
        rng = np.random.default_rng(1)
        for i in range(3):
            dm.save(f"q{i}", f"a{i}", rng.standard_normal(4).astype("float32"))

        assert len(evicted) >= 1, "on_evict must fire when 3rd save overflows cap=2"
        assert plugin.num_entries <= 2, (
            f"cache must respect maxsize=2 after overflow, "
            f"got {plugin.num_entries}"
        )
        # The evicted key(s) must be valid ids that once existed.
        for e in evicted:
            assert isinstance(e, int) and e >= 1

    def test_cost_aware_eviction_via_metadata_callback(self):
        """metadata_callback lets GDSF see per-entry (size, cost).

        Given a cache of capacity 3 and 4 entries with costs [0.01, 1.0, 10.0,
        0.5], GDSF must not evict the $10 entry -- the most expensive entry
        must survive under cost-aware eviction.
        """
        # SSDataManager assigns SQLite AUTOINCREMENT ids monotonically from
        # 1, so we can pre-populate the metadata lookup keyed by predicted id.
        id_to_cost = {}

        def cb(key):
            return id_to_cost.get(key, (1, 1.0))

        evicted = []
        _, _, plugin, dm = _make_stack(
            3, metadata_cb=cb, on_evict=lambda k: evicted.extend(k)
        )
        rng = np.random.default_rng(2)
        costs = [0.01, 1.0, 10.0, 0.5]  # 4 entries, cap=3, so one must go

        # Pre-populate the cost table using the id that SQLite will assign.
        for predicted_id, c in enumerate(costs, start=1):
            id_to_cost[predicted_id] = (1, c)  # size=1 per entry

        for i, c in enumerate(costs):
            dm.save(f"q{i}", f"a{i}", rng.standard_normal(4).astype("float32"))

        # After 4 saves with cap=3, at least one eviction must have happened.
        assert len(evicted) >= 1, (
            f"cap=3 with 4 saves must evict, got evicted={evicted}"
        )
        # The $10 entry must not be evicted (cost-aware invariant).
        expensive_ids = [k for k, (_, cost) in id_to_cost.items() if cost >= 10.0]
        assert expensive_ids, "test setup: at least one expensive id expected"
        for exp_id in expensive_ids:
            assert exp_id not in evicted, (
                f"cost-aware GDSF must NOT evict expensive entry {exp_id} "
                f"(cost=$10); evicted={evicted}"
            )
            assert exp_id in plugin._manager._metadata, (
                f"expensive entry {exp_id} must remain in the cache"
            )
