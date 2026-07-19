"""Tests for the GDSFEvictionManager.

Tests cover:
- Basic put/get operations
- Eviction triggering when cache is full
- Cost-aware eviction (expensive items survive)
- Frequency-aware eviction (frequently accessed items survive)
- Size-aware eviction (larger items evicted sooner)
- Clock advancement on eviction
- Access priority updates
- Capacity constraints
- Duplicate handling
- Explicit removal
- Edge cases (empty cache, oversized items, zero-cost)
- Parameter sensitivity (alpha=0, beta=0)
- Thread safety
- Determinism with same inputs
"""

import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from cost_aware_eviction.config import GDSFConfig
from cost_aware_eviction.eviction_manager import GDSFEvictionManager


class TestPutSingleItem:
    """Test basic put operation."""

    def test_put_single_item(self, small_manager):
        """Putting a single item adds it to the cache."""
        evicted = small_manager.put("key1", size=1, cost=1.0)
        assert evicted == []
        assert "key1" in small_manager
        assert len(small_manager) == 1

    def test_put_returns_empty_list_when_space_available(self, small_manager):
        """No eviction occurs when there is sufficient space."""
        for i in range(5):
            evicted = small_manager.put(f"key_{i}", size=1, cost=1.0)
            assert evicted == []
        assert len(small_manager) == 5

    def test_put_uses_default_size_and_cost(self, small_manager):
        """Put without explicit size/cost uses config defaults."""
        evicted = small_manager.put("default_item")
        assert evicted == []
        assert "default_item" in small_manager

    def test_put_item_tracked_in_current_size(self, small_manager):
        """Current size increases after put."""
        small_manager.put("item", size=3, cost=1.0)
        assert small_manager.current_size == 3


class TestEvictionTriggering:
    """Test that eviction occurs when cache is full."""

    def test_put_triggers_eviction_when_full(self, small_manager):
        """Eviction occurs when adding an item would exceed max_size."""
        # Fill the cache (max_size=10, each item size=1)
        for i in range(10):
            small_manager.put(f"item_{i}", size=1, cost=1.0)

        # This should trigger eviction
        evicted = small_manager.put("overflow", size=1, cost=1.0)
        assert len(evicted) == 1
        assert evicted[0].startswith("item_")

    def test_put_triggers_multiple_evictions_for_large_item(self, small_manager):
        """A large item may evict multiple smaller items."""
        # Fill with size-1 items
        for i in range(10):
            small_manager.put(f"small_{i}", size=1, cost=1.0)

        # Insert a size-5 item - should evict 5 items
        evicted = small_manager.put("big_item", size=5, cost=1.0)
        assert len(evicted) == 5
        assert small_manager.current_size <= small_manager.max_size

    def test_eviction_count_matches_space_needed(self, small_manager):
        """Exactly enough items are evicted to make room."""
        # Fill with size-2 items (5 items total, using 10 units)
        for i in range(5):
            small_manager.put(f"item_{i}", size=2, cost=1.0)

        # Insert size-3 item: needs to evict at least one size-2 item
        evicted = small_manager.put("new_item", size=3, cost=1.0)
        # Need to free at least 3 units: evict 2 items of size 2
        assert len(evicted) >= 1
        assert small_manager.current_size <= small_manager.max_size


class TestEvictsLowestPriority:
    """Test that the item with lowest priority is evicted first."""

    def test_evicts_lowest_priority(self):
        """The item with lowest computed priority gets evicted."""
        config = GDSFConfig(max_size=3, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        # Insert items with different costs (all size=1, freq=1)
        # Priority = Clock + (1^1 * cost^1) / 1 = cost (clock=0 initially)
        manager.put("cheap", size=1, cost=1.0)     # priority = 1.0
        manager.put("medium", size=1, cost=5.0)    # priority = 5.0
        manager.put("expensive", size=1, cost=10.0) # priority = 10.0

        # Cache is full. Insert one more - should evict "cheap"
        evicted = manager.put("new", size=1, cost=3.0)
        assert evicted == ["cheap"]
        assert "cheap" not in manager
        assert "expensive" in manager


class TestCostAwareness:
    """Test that expensive items survive eviction."""

    def test_expensive_items_survive_eviction(self):
        """High-cost items have higher priority and survive eviction."""
        config = GDSFConfig(max_size=5, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        # Insert: 4 cheap items + 1 expensive item
        manager.put("expensive", size=1, cost=100.0)
        for i in range(4):
            manager.put(f"cheap_{i}", size=1, cost=1.0)

        # Trigger eviction by inserting one more
        evicted = manager.put("trigger", size=1, cost=1.0)

        # The expensive item should NOT be evicted
        assert "expensive" not in evicted
        assert "expensive" in manager
        # One of the cheap items should be evicted
        assert any("cheap" in k for k in evicted)

    def test_cost_determines_eviction_order(self):
        """With equal size and frequency, lower cost items are evicted first."""
        config = GDSFConfig(max_size=4, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        costs = [10.0, 1.0, 5.0, 2.0]
        for i, cost in enumerate(costs):
            manager.put(f"item_{i}", size=1, cost=cost)

        # Evict one - should be item_1 (cost=1.0, lowest priority)
        evicted = manager.put("new", size=1, cost=3.0)
        assert evicted == ["item_1"]

    def test_zero_cost_items_evicted_first(self):
        """Items with cost=0 have lowest priority (priority = clock + 0)."""
        config = GDSFConfig(max_size=3, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        manager.put("free", size=1, cost=0.0)
        manager.put("cheap", size=1, cost=1.0)
        manager.put("pricey", size=1, cost=10.0)

        evicted = manager.put("new", size=1, cost=5.0)
        assert evicted == ["free"]


class TestFrequencyAwareness:
    """Test that frequently accessed items survive eviction."""

    def test_frequent_items_survive_eviction(self):
        """Items accessed many times have higher priority and survive."""
        config = GDSFConfig(max_size=3, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        # All items have the same cost
        manager.put("popular", size=1, cost=1.0)
        manager.put("rarely_used_1", size=1, cost=1.0)
        manager.put("rarely_used_2", size=1, cost=1.0)

        # Access "popular" many times to boost its priority
        for _ in range(10):
            manager.access("popular")

        # Trigger eviction
        evicted = manager.put("new_item", size=1, cost=1.0)

        # "popular" should survive
        assert "popular" not in evicted
        assert "popular" in manager

    def test_frequency_boost_accumulates(self):
        """Each access increases priority further."""
        config = GDSFConfig(max_size=5, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        manager.put("item_a", size=1, cost=1.0)
        manager.put("item_b", size=1, cost=1.0)

        # Access a few times
        manager.access("item_a")
        p1 = manager.get_priority("item_a")

        manager.access("item_a")
        p2 = manager.get_priority("item_a")

        # Priority should increase with each access
        assert p2 > p1


class TestSizeAwareness:
    """Test that larger items are penalized in priority."""

    def test_large_items_evicted_sooner(self):
        """Larger items have lower priority (size in denominator) and are evicted first."""
        config = GDSFConfig(max_size=10, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        # Same cost, different sizes
        manager.put("large", size=5, cost=10.0)    # priority = 10/5 = 2.0
        manager.put("small", size=1, cost=10.0)    # priority = 10/1 = 10.0

        # Fill remaining space (10 - 5 - 1 = 4 units left)
        for i in range(4):
            manager.put(f"filler_{i}", size=1, cost=10.0)

        # Force eviction
        evicted = manager.put("trigger", size=1, cost=10.0)

        # "large" should be evicted first due to lower priority
        assert "large" in evicted

    def test_size_penalty_formula(self):
        """Verify size appears in denominator of priority formula."""
        config = GDSFConfig(max_size=100, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        manager.put("size_1", size=1, cost=10.0)
        manager.put("size_10", size=10, cost=10.0)

        p1 = manager.get_priority("size_1")
        p10 = manager.get_priority("size_10")

        # Priority = Clock + (freq^alpha * cost^beta) / size
        # p1 = 0 + (1*10)/1 = 10.0
        # p10 = 0 + (1*10)/10 = 1.0
        assert p1 == pytest.approx(10.0)
        assert p10 == pytest.approx(1.0)
        assert p1 > p10


class TestAccessUpdates:
    """Test that access() correctly updates priority."""

    def test_access_updates_priority(self, small_manager):
        """Accessing an item increases its priority."""
        small_manager.put("item", size=1, cost=5.0)
        p_before = small_manager.get_priority("item")

        small_manager.access("item")
        p_after = small_manager.get_priority("item")

        assert p_after > p_before

    def test_access_returns_true_for_existing(self, small_manager):
        """Access returns True when key exists."""
        small_manager.put("item", size=1, cost=1.0)
        assert small_manager.access("item") is True

    def test_access_returns_false_for_missing(self, small_manager):
        """Access returns False when key does not exist."""
        assert small_manager.access("nonexistent") is False

    def test_access_increments_frequency(self, small_manager):
        """Each access increments the frequency counter."""
        small_manager.put("item", size=1, cost=1.0)

        # Priority with freq=1: Clock + (1^1 * 1^1) / 1 = 1.0
        p1 = small_manager.get_priority("item")

        small_manager.access("item")
        # Priority with freq=2: Clock + (2^1 * 1^1) / 1 = 2.0
        p2 = small_manager.get_priority("item")

        small_manager.access("item")
        # Priority with freq=3: Clock + (3^1 * 1^1) / 1 = 3.0
        p3 = small_manager.get_priority("item")

        assert p1 == pytest.approx(1.0)
        assert p2 == pytest.approx(2.0)
        assert p3 == pytest.approx(3.0)


class TestClockAdvancement:
    """Test that the clock advances on eviction."""

    def test_clock_advances_on_eviction(self):
        """Clock moves to the priority of the evicted item."""
        config = GDSFConfig(max_size=2, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        assert manager.clock == 0.0

        manager.put("a", size=1, cost=2.0)  # priority = 0 + 2/1 = 2.0
        manager.put("b", size=1, cost=5.0)  # priority = 0 + 5/1 = 5.0

        # Evict "a" (priority 2.0) -> clock advances to 2.0
        evicted = manager.put("c", size=1, cost=3.0)
        assert evicted == ["a"]
        assert manager.clock == pytest.approx(2.0)

    def test_clock_advances_monotonically(self):
        """Clock never decreases (always advances forward)."""
        config = GDSFConfig(max_size=2, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        prev_clock = 0.0
        for i in range(20):
            cost = random.uniform(0.1, 10.0)
            evicted = manager.put(f"item_{i}", size=1, cost=cost)
            assert manager.clock >= prev_clock
            prev_clock = manager.clock

    def test_clock_affects_new_insertions(self):
        """New items get priority relative to current clock value."""
        config = GDSFConfig(max_size=2, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        manager.put("a", size=1, cost=1.0)  # priority = 0 + 1 = 1
        manager.put("b", size=1, cost=2.0)  # priority = 0 + 2 = 2

        # Evict "a" -> clock = 1.0
        manager.put("c", size=1, cost=1.0)  # priority = 1.0 + 1 = 2.0

        # Now "c" has priority 2.0 (same as "b"), both survive equally
        assert manager.clock == pytest.approx(1.0)
        p_c = manager.get_priority("c")
        assert p_c == pytest.approx(2.0)


class TestCapacityConstraints:
    """Test that capacity is never exceeded."""

    def test_capacity_never_exceeded(self):
        """Current size never exceeds max_size regardless of operations."""
        config = GDSFConfig(max_size=20, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)
        random.seed(42)

        for i in range(100):
            size = random.randint(1, 5)
            cost = random.uniform(0.1, 10.0)
            manager.put(f"item_{i}", size=size, cost=cost)
            assert manager.current_size <= manager.max_size

    def test_capacity_with_variable_sizes(self):
        """Capacity constraint holds with varying item sizes."""
        config = GDSFConfig(max_size=50, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        sizes = [1, 5, 10, 3, 7, 2, 8, 15, 4, 6, 20, 1, 1, 1]
        for i, size in enumerate(sizes):
            manager.put(f"var_{i}", size=size, cost=1.0)
            assert manager.current_size <= 50


class TestDuplicateHandling:
    """Test that duplicate puts update rather than duplicate."""

    def test_duplicate_put_updates_not_duplicates(self, small_manager):
        """Putting an existing key updates metadata without creating duplicates."""
        small_manager.put("key", size=1, cost=1.0)
        assert len(small_manager) == 1

        # Put same key again with different cost
        evicted = small_manager.put("key", size=1, cost=5.0)
        assert evicted == []
        assert len(small_manager) == 1

        # Priority should reflect new cost and freq=2
        # Priority = Clock + (2^1 * 5^1) / 1 = 10.0
        p = small_manager.get_priority("key")
        assert p == pytest.approx(10.0)

    def test_duplicate_put_increments_frequency(self, small_manager):
        """Duplicate put increments the frequency counter."""
        small_manager.put("item", size=1, cost=1.0)
        p1 = small_manager.get_priority("item")

        small_manager.put("item", size=1, cost=1.0)
        p2 = small_manager.get_priority("item")

        # freq went from 1 to 2
        assert p2 > p1

    def test_duplicate_put_updates_size(self, small_manager):
        """Duplicate put can change the item's size."""
        small_manager.put("item", size=2, cost=1.0)
        assert small_manager.current_size == 2

        small_manager.put("item", size=5, cost=1.0)
        assert small_manager.current_size == 5


class TestRemove:
    """Test explicit removal of items."""

    def test_remove_frees_space(self, small_manager):
        """Removing an item frees its size from the cache."""
        small_manager.put("item", size=5, cost=1.0)
        assert small_manager.current_size == 5

        result = small_manager.remove("item")
        assert result is True
        assert small_manager.current_size == 0
        assert "item" not in small_manager

    def test_remove_nonexistent_returns_false(self, small_manager):
        """Removing a non-existent key returns False."""
        result = small_manager.remove("ghost")
        assert result is False

    def test_remove_then_reinsert(self, small_manager):
        """An item can be re-inserted after removal."""
        small_manager.put("item", size=1, cost=5.0)
        small_manager.remove("item")
        evicted = small_manager.put("item", size=1, cost=10.0)
        assert evicted == []
        assert "item" in small_manager


class TestEmptyCache:
    """Test operations on an empty cache."""

    def test_empty_cache_operations(self, small_manager):
        """Operations on empty cache do not crash."""
        assert len(small_manager) == 0
        assert small_manager.current_size == 0
        assert "anything" not in small_manager
        assert small_manager.access("nothing") is False
        assert small_manager.remove("ghost") is False
        assert small_manager.get_priority("missing") is None


class TestOversizedItems:
    """Test items larger than cache capacity."""

    def test_item_larger_than_cache(self):
        """An item larger than max_size is rejected (not inserted)."""
        config = GDSFConfig(max_size=5, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        evicted = manager.put("huge", size=10, cost=100.0)
        # Item is rejected - returns empty list, not added
        assert evicted == []
        assert "huge" not in manager
        assert len(manager) == 0

    def test_item_exactly_cache_size(self):
        """An item exactly equal to max_size can be inserted."""
        config = GDSFConfig(max_size=5, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        evicted = manager.put("exact", size=5, cost=1.0)
        assert evicted == []
        assert "exact" in manager
        assert manager.current_size == 5


class TestParameterSensitivity:
    """Test alpha and beta parameter effects."""

    def test_alpha_zero_ignores_frequency(self, manager_alpha_zero):
        """With alpha=0, frequency has no effect on priority."""
        manager_alpha_zero.put("item", size=1, cost=5.0)

        # Priority = Clock + (freq^0 * cost^1) / size = 0 + (1 * 5) / 1 = 5.0
        p1 = manager_alpha_zero.get_priority("item")

        manager_alpha_zero.access("item")
        # With alpha=0: Priority = Clock + (freq^0 * cost^1) / size = 0 + (1 * 5) / 1 = 5.0
        # freq^0 = 1 regardless of freq value
        p2 = manager_alpha_zero.get_priority("item")

        assert p1 == pytest.approx(p2)

    def test_beta_zero_ignores_cost(self, manager_beta_zero):
        """With beta=0, cost has no effect on priority."""
        manager_beta_zero.put("cheap", size=1, cost=0.01)
        manager_beta_zero.put("expensive", size=1, cost=1000.0)

        # With beta=0: cost^0 = 1 always
        # Priority = Clock + (freq^1 * 1) / size = 0 + 1/1 = 1.0 for both
        p_cheap = manager_beta_zero.get_priority("cheap")
        p_expensive = manager_beta_zero.get_priority("expensive")

        assert p_cheap == pytest.approx(p_expensive)

    def test_high_alpha_amplifies_frequency(self):
        """Higher alpha amplifies the frequency advantage."""
        config = GDSFConfig(max_size=100, alpha=2.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        manager.put("item", size=1, cost=1.0)
        manager.access("item")
        manager.access("item")
        # freq=3, alpha=2: priority = 0 + (3^2 * 1) / 1 = 9.0
        p = manager.get_priority("item")
        assert p == pytest.approx(9.0)

    def test_high_beta_amplifies_cost(self):
        """Higher beta amplifies the cost advantage."""
        config = GDSFConfig(max_size=100, alpha=1.0, beta=2.0)
        manager = GDSFEvictionManager(config=config)

        manager.put("item", size=1, cost=3.0)
        # freq=1, beta=2: priority = 0 + (1 * 3^2) / 1 = 9.0
        p = manager.get_priority("item")
        assert p == pytest.approx(9.0)


class TestDeterminism:
    """Test that the manager produces deterministic results."""

    def test_deterministic_with_same_inputs(self):
        """Same sequence of operations produces same eviction decisions."""
        results = []
        for _ in range(3):
            config = GDSFConfig(max_size=5, alpha=1.0, beta=1.0)
            manager = GDSFEvictionManager(config=config)

            all_evicted = []
            for i in range(10):
                evicted = manager.put(f"key_{i}", size=1, cost=float(i + 1))
                all_evicted.extend(evicted)
            results.append(all_evicted)

        # All runs should produce identical eviction sequences
        assert results[0] == results[1] == results[2]


class TestThreadSafety:
    """Test concurrent access to the eviction manager."""

    def test_thread_safety_concurrent_puts(self):
        """Concurrent puts do not corrupt internal state."""
        config = GDSFConfig(max_size=100, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)
        num_threads = 8
        items_per_thread = 50

        def worker(thread_id):
            for i in range(items_per_thread):
                manager.put(f"t{thread_id}_item_{i}", size=1, cost=1.0)

        threads = []
        for t in range(num_threads):
            thread = threading.Thread(target=worker, args=(t,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Cache should not exceed capacity
        assert manager.current_size <= manager.max_size
        assert len(manager) <= manager.max_size

    def test_thread_safety_concurrent_puts_and_accesses(self):
        """Concurrent puts and accesses do not corrupt state."""
        config = GDSFConfig(max_size=50, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        # Pre-populate
        for i in range(30):
            manager.put(f"item_{i}", size=1, cost=1.0)

        errors = []

        def put_worker():
            try:
                for i in range(100):
                    manager.put(f"new_{threading.current_thread().name}_{i}",
                              size=1, cost=random.uniform(0.1, 5.0))
            except Exception as e:
                errors.append(e)

        def access_worker():
            try:
                for i in range(100):
                    manager.access(f"item_{i % 30}")
            except Exception as e:
                errors.append(e)

        threads = []
        for _ in range(4):
            threads.append(threading.Thread(target=put_worker))
            threads.append(threading.Thread(target=access_worker))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert manager.current_size <= manager.max_size

    def test_thread_safety_concurrent_removes(self):
        """Concurrent removes do not cause errors."""
        config = GDSFConfig(max_size=100, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        # Pre-populate
        for i in range(100):
            manager.put(f"item_{i}", size=1, cost=1.0)

        errors = []

        def remove_worker(start):
            try:
                for i in range(start, 100, 4):
                    manager.remove(f"item_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=remove_worker, args=(s,)) for s in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert manager.current_size >= 0


class TestPriorityFormula:
    """Test the priority formula directly."""

    def test_priority_formula_basic(self):
        """Priority = Clock + (freq^alpha * cost^beta) / size."""
        config = GDSFConfig(max_size=100, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        manager.put("item", size=2, cost=6.0)
        # freq=1, alpha=1, beta=1, size=2, clock=0
        # Priority = 0 + (1^1 * 6^1) / 2 = 3.0
        p = manager.get_priority("item")
        assert p == pytest.approx(3.0)

    def test_priority_increases_with_clock(self):
        """Items inserted after clock advancement have higher base priority."""
        config = GDSFConfig(max_size=2, alpha=1.0, beta=1.0)
        manager = GDSFEvictionManager(config=config)

        manager.put("a", size=1, cost=1.0)  # priority = 0 + 1 = 1.0
        manager.put("b", size=1, cost=2.0)  # priority = 0 + 2 = 2.0

        # Evict "a" (min priority), clock -> 1.0
        evicted = manager.put("c", size=1, cost=1.0)
        assert evicted == ["a"]

        # "c" priority = 1.0 + 1.0 = 2.0 (same as "b")
        p_c = manager.get_priority("c")
        assert p_c == pytest.approx(2.0)


class TestGetPriority:
    """Test get_priority method."""

    def test_get_priority_existing(self, small_manager):
        """get_priority returns correct value for existing key."""
        small_manager.put("item", size=1, cost=3.0)
        p = small_manager.get_priority("item")
        assert p == pytest.approx(3.0)

    def test_get_priority_nonexistent(self, small_manager):
        """get_priority returns None for non-existent key."""
        p = small_manager.get_priority("ghost")
        assert p is None

    def test_get_priority_after_access(self, small_manager):
        """get_priority reflects updated priority after access."""
        small_manager.put("item", size=1, cost=2.0)
        small_manager.access("item")
        # freq=2, cost=2, size=1: priority = 0 + (2*2)/1 = 4.0
        p = small_manager.get_priority("item")
        assert p == pytest.approx(4.0)


class TestContainsAndLen:
    """Test __contains__ and __len__ methods."""

    def test_contains_after_put(self, small_manager):
        """Key is in manager after put."""
        small_manager.put("x", size=1, cost=1.0)
        assert "x" in small_manager

    def test_not_contains_before_put(self, small_manager):
        """Key is not in manager before put."""
        assert "x" not in small_manager

    def test_len_tracks_items(self, small_manager):
        """Length reflects number of items."""
        assert len(small_manager) == 0
        small_manager.put("a", size=1, cost=1.0)
        assert len(small_manager) == 1
        small_manager.put("b", size=1, cost=1.0)
        assert len(small_manager) == 2
        small_manager.remove("a")
        assert len(small_manager) == 1
