"""Tests for the IndexedMinHeap priority queue.

Tests cover:
- Basic push/pop ordering
- Priority updates (increase and decrease)
- Element removal from arbitrary positions
- Peek operation
- Contains check
- Length tracking
- Error handling for empty heap and duplicate keys
- Large heap correctness (100+ elements)
- Stress testing with random operations (1000 ops)
"""

import random

import pytest

from cost_aware_eviction.priority_queue import IndexedMinHeap


class TestPushAndPopOrder:
    """Test that pop always returns the minimum priority element."""

    def test_push_and_pop_order_ascending(self, empty_heap):
        """Items pushed in ascending priority order are popped in same order."""
        empty_heap.push("a", 1.0)
        empty_heap.push("b", 2.0)
        empty_heap.push("c", 3.0)

        key1, pri1 = empty_heap.pop()
        key2, pri2 = empty_heap.pop()
        key3, pri3 = empty_heap.pop()

        assert (key1, pri1) == ("a", 1.0)
        assert (key2, pri2) == ("b", 2.0)
        assert (key3, pri3) == ("c", 3.0)

    def test_push_and_pop_order_descending(self, empty_heap):
        """Items pushed in descending priority order are still popped min-first."""
        empty_heap.push("c", 3.0)
        empty_heap.push("b", 2.0)
        empty_heap.push("a", 1.0)

        key1, pri1 = empty_heap.pop()
        key2, pri2 = empty_heap.pop()
        key3, pri3 = empty_heap.pop()

        assert (key1, pri1) == ("a", 1.0)
        assert (key2, pri2) == ("b", 2.0)
        assert (key3, pri3) == ("c", 3.0)

    def test_push_and_pop_order_random(self, empty_heap):
        """Items pushed in random order are popped in sorted priority order."""
        items = [("x", 5.0), ("y", 1.0), ("z", 3.0), ("w", 2.0), ("v", 4.0)]
        for key, priority in items:
            empty_heap.push(key, priority)

        popped = []
        while len(empty_heap) > 0:
            popped.append(empty_heap.pop())

        priorities = [p for _, p in popped]
        assert priorities == sorted(priorities)

    def test_push_and_pop_single_item(self, empty_heap):
        """A single pushed item can be popped correctly."""
        empty_heap.push("only", 42.0)
        key, priority = empty_heap.pop()
        assert key == "only"
        assert priority == 42.0
        assert len(empty_heap) == 0


class TestUpdatePriority:
    """Test priority update operations."""

    def test_update_priority_decrease(self, populated_heap):
        """Decreasing priority of a non-min element makes it the new minimum."""
        # "e" has priority 5.0; update to 0.5 (lower than "a" at 1.0)
        populated_heap.update("e", 0.5)

        key, priority = populated_heap.peek()
        assert key == "e"
        assert priority == 0.5

    def test_update_priority_increase(self, populated_heap):
        """Increasing priority of the min element changes the minimum."""
        # "a" has priority 1.0; increase to 10.0
        populated_heap.update("a", 10.0)

        key, priority = populated_heap.peek()
        # Now "b" (priority 2.0) should be the minimum
        assert key == "b"
        assert priority == 2.0

    def test_update_priority_no_change(self, populated_heap):
        """Updating to the same priority leaves heap unchanged."""
        populated_heap.update("c", 3.0)  # Same priority
        # Verify heap is still valid by popping all in order
        priorities = []
        while len(populated_heap) > 0:
            _, p = populated_heap.pop()
            priorities.append(p)
        assert priorities == sorted(priorities)

    def test_update_nonexistent_key_raises(self, populated_heap):
        """Updating a non-existent key raises KeyError."""
        with pytest.raises(KeyError):
            populated_heap.update("nonexistent", 1.0)

    def test_update_makes_element_new_max(self, populated_heap):
        """Increasing priority of an element to the max still maintains order."""
        populated_heap.update("a", 100.0)

        # Pop all and verify order
        results = []
        while len(populated_heap) > 0:
            results.append(populated_heap.pop())

        priorities = [p for _, p in results]
        assert priorities == sorted(priorities)
        assert results[-1] == ("a", 100.0)


class TestRemove:
    """Test removal of elements from the heap."""

    def test_remove_middle_element(self, populated_heap):
        """Removing a middle element maintains heap integrity."""
        removed_priority = populated_heap.remove("c")
        assert removed_priority == 3.0
        assert "c" not in populated_heap
        assert len(populated_heap) == 4

        # Verify remaining elements pop in correct order
        results = []
        while len(populated_heap) > 0:
            results.append(populated_heap.pop())
        priorities = [p for _, p in results]
        assert priorities == sorted(priorities)
        assert all(k != "c" for k, _ in results)

    def test_remove_minimum_element(self, populated_heap):
        """Removing the minimum element makes the next smallest the new min."""
        populated_heap.remove("a")
        key, priority = populated_heap.peek()
        assert key == "b"
        assert priority == 2.0

    def test_remove_maximum_element(self, populated_heap):
        """Removing the maximum element does not affect minimum."""
        populated_heap.remove("e")
        key, priority = populated_heap.peek()
        assert key == "a"
        assert priority == 1.0
        assert len(populated_heap) == 4

    def test_remove_last_element(self, empty_heap):
        """Removing the only element leaves an empty heap."""
        empty_heap.push("sole", 7.0)
        empty_heap.remove("sole")
        assert len(empty_heap) == 0
        assert "sole" not in empty_heap

    def test_remove_nonexistent_key_raises(self, populated_heap):
        """Removing a non-existent key raises KeyError."""
        with pytest.raises(KeyError):
            populated_heap.remove("nonexistent")

    def test_remove_all_elements_one_by_one(self, populated_heap):
        """Removing all elements one by one empties the heap."""
        for key in ["c", "a", "e", "b", "d"]:
            populated_heap.remove(key)
        assert len(populated_heap) == 0


class TestPeek:
    """Test the peek operation."""

    def test_peek_returns_minimum(self, populated_heap):
        """Peek returns the minimum priority element without removing it."""
        key, priority = populated_heap.peek()
        assert key == "a"
        assert priority == 1.0
        # Verify element is still in the heap
        assert len(populated_heap) == 5
        assert "a" in populated_heap

    def test_peek_empty_heap_raises(self, empty_heap):
        """Peeking at an empty heap raises IndexError."""
        with pytest.raises(IndexError):
            empty_heap.peek()

    def test_peek_after_pop(self, populated_heap):
        """Peek reflects new minimum after pop."""
        populated_heap.pop()  # removes "a"
        key, priority = populated_heap.peek()
        assert key == "b"
        assert priority == 2.0


class TestContains:
    """Test the __contains__ (in) operator."""

    def test_contains_existing_key(self, populated_heap):
        """An existing key is found in the heap."""
        assert "a" in populated_heap
        assert "c" in populated_heap
        assert "e" in populated_heap

    def test_contains_nonexistent_key(self, populated_heap):
        """A non-existent key is not found in the heap."""
        assert "z" not in populated_heap
        assert "nonexistent" not in populated_heap

    def test_contains_after_removal(self, populated_heap):
        """A removed key is no longer found in the heap."""
        populated_heap.remove("c")
        assert "c" not in populated_heap

    def test_contains_after_pop(self, populated_heap):
        """A popped key is no longer found in the heap."""
        key, _ = populated_heap.pop()
        assert key not in populated_heap


class TestLen:
    """Test the __len__ operation."""

    def test_len_empty(self, empty_heap):
        """An empty heap has length 0."""
        assert len(empty_heap) == 0

    def test_len_after_pushes(self, populated_heap):
        """Length increases with pushes."""
        assert len(populated_heap) == 5

    def test_len_after_pop(self, populated_heap):
        """Length decreases after pop."""
        populated_heap.pop()
        assert len(populated_heap) == 4

    def test_len_after_remove(self, populated_heap):
        """Length decreases after remove."""
        populated_heap.remove("c")
        assert len(populated_heap) == 4

    def test_len_push_pop_cycle(self, empty_heap):
        """Length tracks correctly through push/pop cycles."""
        empty_heap.push("x", 1.0)
        assert len(empty_heap) == 1
        empty_heap.push("y", 2.0)
        assert len(empty_heap) == 2
        empty_heap.pop()
        assert len(empty_heap) == 1
        empty_heap.pop()
        assert len(empty_heap) == 0


class TestErrorHandling:
    """Test error cases."""

    def test_empty_pop_raises(self, empty_heap):
        """Popping from an empty heap raises IndexError."""
        with pytest.raises(IndexError, match="pop from empty heap"):
            empty_heap.pop()

    def test_duplicate_key_raises(self, populated_heap):
        """Pushing a duplicate key raises KeyError."""
        with pytest.raises(KeyError, match="already exists"):
            populated_heap.push("a", 99.0)

    def test_update_empty_heap_raises(self, empty_heap):
        """Updating in an empty heap raises KeyError."""
        with pytest.raises(KeyError):
            empty_heap.update("ghost", 1.0)

    def test_remove_from_empty_heap_raises(self, empty_heap):
        """Removing from an empty heap raises KeyError."""
        with pytest.raises(KeyError):
            empty_heap.remove("ghost")


class TestLargeHeap:
    """Test with larger data volumes."""

    def test_large_heap_ordering(self):
        """100+ elements are popped in correct min-priority order."""
        heap = IndexedMinHeap()
        n = 150
        priorities = list(range(n))
        random.seed(42)
        random.shuffle(priorities)

        for i, p in enumerate(priorities):
            heap.push(f"key_{i}", float(p))

        assert len(heap) == n

        prev_priority = float("-inf")
        for _ in range(n):
            key, priority = heap.pop()
            assert priority >= prev_priority
            prev_priority = priority

        assert len(heap) == 0

    def test_large_heap_with_updates(self):
        """Large heap maintains order after many updates."""
        heap = IndexedMinHeap()
        n = 200
        random.seed(123)

        # Push elements
        for i in range(n):
            heap.push(f"item_{i}", random.uniform(0, 1000))

        # Perform 100 random updates
        for _ in range(100):
            idx = random.randint(0, n - 1)
            new_priority = random.uniform(0, 1000)
            heap.update(f"item_{idx}", new_priority)

        # Verify pop order
        prev = float("-inf")
        while len(heap) > 0:
            _, priority = heap.pop()
            assert priority >= prev
            prev = priority


class TestStressRandomOperations:
    """Stress test with 1000 random operations."""

    def test_stress_random_operations(self):
        """1000 random push/pop/update/remove operations maintain heap invariant."""
        heap = IndexedMinHeap()
        reference = {}  # key -> priority (ground truth)
        random.seed(2024)
        next_key_id = 0

        for _ in range(1000):
            op = random.choice(["push", "pop", "update", "remove", "peek"])

            if op == "push" or (not reference and op != "push"):
                key = f"k_{next_key_id}"
                next_key_id += 1
                priority = random.uniform(-100, 100)
                heap.push(key, priority)
                reference[key] = priority

            elif op == "pop" and reference:
                key, priority = heap.pop()
                assert key in reference
                # The popped item should have the minimum priority
                min_priority = min(reference.values())
                assert abs(priority - min_priority) < 1e-10, (
                    f"Popped priority {priority} != min {min_priority}"
                )
                del reference[key]

            elif op == "update" and reference:
                key = random.choice(list(reference.keys()))
                new_priority = random.uniform(-100, 100)
                heap.update(key, new_priority)
                reference[key] = new_priority

            elif op == "remove" and reference:
                key = random.choice(list(reference.keys()))
                heap.remove(key)
                del reference[key]

            elif op == "peek" and reference:
                key, priority = heap.peek()
                min_priority = min(reference.values())
                assert abs(priority - min_priority) < 1e-10

            # Invariant: lengths must always match
            assert len(heap) == len(reference)

        # Final drain: verify all remaining elements pop in order
        prev_priority = float("-inf")
        while reference:
            key, priority = heap.pop()
            assert priority >= prev_priority - 1e-10
            assert key in reference
            del reference[key]
            prev_priority = priority

        assert len(heap) == 0

    def test_stress_many_pops(self):
        """Push many, then pop all - verifies no corruption under repeated pops."""
        heap = IndexedMinHeap()
        random.seed(999)
        n = 500
        expected_priorities = []

        for i in range(n):
            p = random.uniform(0, 10000)
            heap.push(f"stress_{i}", p)
            expected_priorities.append(p)

        expected_priorities.sort()
        actual_priorities = []
        for _ in range(n):
            _, p = heap.pop()
            actual_priorities.append(p)

        for expected, actual in zip(expected_priorities, actual_priorities):
            assert abs(expected - actual) < 1e-10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_negative_priorities(self, empty_heap):
        """Negative priorities are handled correctly."""
        empty_heap.push("neg", -5.0)
        empty_heap.push("pos", 5.0)
        empty_heap.push("zero", 0.0)

        key, priority = empty_heap.pop()
        assert key == "neg"
        assert priority == -5.0

    def test_equal_priorities(self, empty_heap):
        """Elements with equal priorities can coexist."""
        empty_heap.push("x", 1.0)
        empty_heap.push("y", 1.0)
        empty_heap.push("z", 1.0)

        assert len(empty_heap) == 3
        _, p1 = empty_heap.pop()
        _, p2 = empty_heap.pop()
        _, p3 = empty_heap.pop()
        assert p1 == p2 == p3 == 1.0

    def test_very_large_priorities(self, empty_heap):
        """Very large priority values are handled correctly."""
        empty_heap.push("big", 1e18)
        empty_heap.push("small", -1e18)
        key, priority = empty_heap.pop()
        assert key == "small"
        assert priority == -1e18

    def test_float_precision(self, empty_heap):
        """Close float priorities maintain relative order."""
        empty_heap.push("a", 1.0000000001)
        empty_heap.push("b", 1.0000000002)
        empty_heap.push("c", 1.0)

        key, _ = empty_heap.pop()
        assert key == "c"

    def test_various_key_types(self):
        """Heap works with different hashable key types."""
        heap = IndexedMinHeap()
        heap.push(1, 3.0)
        heap.push("str_key", 1.0)
        heap.push((1, 2), 2.0)

        key, _ = heap.pop()
        assert key == "str_key"
        key, _ = heap.pop()
        assert key == (1, 2)
        key, _ = heap.pop()
        assert key == 1
