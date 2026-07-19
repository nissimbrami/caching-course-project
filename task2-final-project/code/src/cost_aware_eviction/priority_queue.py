"""Indexed min-heap (priority queue) with O(log n) update and remove operations.

This data structure supports:
- push(key, priority): Insert a new key with given priority
- pop(): Remove and return the (key, priority) pair with minimum priority
- update(key, new_priority): Change the priority of an existing key
- remove(key): Remove a specific key from the heap
- peek(): Return the minimum (key, priority) without removing
- __contains__(key): Check if a key exists in the heap
- __len__(): Return number of elements
"""

from typing import Any, Tuple, List, Optional


class IndexedMinHeap:
    """A min-heap with an index for O(log n) priority updates and removals.

    Uses a standard array-based binary heap with an auxiliary dictionary
    mapping keys to their positions in the heap array.
    """

    def __init__(self):
        """Initialize an empty indexed min-heap."""
        self._heap: List[Tuple[float, Any]] = []  # (priority, key) pairs
        self._index: dict = {}  # key -> position in heap

    def __len__(self) -> int:
        """Return the number of elements in the heap."""
        return len(self._heap)

    def __contains__(self, key: Any) -> bool:
        """Check if a key exists in the heap."""
        return key in self._index

    def push(self, key: Any, priority: float) -> None:
        """Insert a new key with the given priority.

        Args:
            key: The key to insert (must be unique).
            priority: The priority value (lower = higher priority).

        Raises:
            KeyError: If the key already exists in the heap.
        """
        if key in self._index:
            raise KeyError(f"Key {key!r} already exists in the heap")

        pos = len(self._heap)
        self._heap.append((priority, key))
        self._index[key] = pos
        self._sift_up(pos)

    def pop(self) -> Tuple[Any, float]:
        """Remove and return the (key, priority) pair with minimum priority.

        Returns:
            Tuple of (key, priority) with the lowest priority value.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self._heap:
            raise IndexError("pop from empty heap")

        # Swap root with last element
        self._swap(0, len(self._heap) - 1)

        # Remove the (now last) minimum element
        priority, key = self._heap.pop()
        del self._index[key]

        # Restore heap property
        if self._heap:
            self._sift_down(0)

        return key, priority

    def peek(self) -> Tuple[Any, float]:
        """Return the (key, priority) pair with minimum priority without removing.

        Returns:
            Tuple of (key, priority) with the lowest priority value.

        Raises:
            IndexError: If the heap is empty.
        """
        if not self._heap:
            raise IndexError("peek at empty heap")
        priority, key = self._heap[0]
        return key, priority

    def update(self, key: Any, new_priority: float) -> None:
        """Update the priority of an existing key.

        Args:
            key: The key whose priority to update.
            new_priority: The new priority value.

        Raises:
            KeyError: If the key does not exist in the heap.
        """
        if key not in self._index:
            raise KeyError(f"Key {key!r} not found in heap")

        pos = self._index[key]
        old_priority = self._heap[pos][0]
        self._heap[pos] = (new_priority, key)

        # Restore heap property
        if new_priority < old_priority:
            self._sift_up(pos)
        elif new_priority > old_priority:
            self._sift_down(pos)

    def remove(self, key: Any) -> float:
        """Remove a specific key from the heap.

        Args:
            key: The key to remove.

        Returns:
            The priority of the removed key.

        Raises:
            KeyError: If the key does not exist in the heap.
        """
        if key not in self._index:
            raise KeyError(f"Key {key!r} not found in heap")

        pos = self._index[key]
        removed_priority = self._heap[pos][0]

        # If it's the last element, just remove it
        if pos == len(self._heap) - 1:
            self._heap.pop()
            del self._index[key]
            return removed_priority

        # Swap with last element and remove
        self._swap(pos, len(self._heap) - 1)
        self._heap.pop()
        del self._index[key]

        # Restore heap property for the swapped element
        if pos < len(self._heap):
            self._sift_up(pos)
            self._sift_down(pos)

        return removed_priority

    def _swap(self, i: int, j: int) -> None:
        """Swap two elements in the heap and update the index."""
        self._heap[i], self._heap[j] = self._heap[j], self._heap[i]
        # Update index
        _, key_i = self._heap[i]
        _, key_j = self._heap[j]
        self._index[key_i] = i
        self._index[key_j] = j

    def _sift_up(self, pos: int) -> None:
        """Move element at pos upward to restore heap property."""
        while pos > 0:
            parent = (pos - 1) // 2
            if self._heap[pos][0] < self._heap[parent][0]:
                self._swap(pos, parent)
                pos = parent
            else:
                break

    def _sift_down(self, pos: int) -> None:
        """Move element at pos downward to restore heap property."""
        n = len(self._heap)
        while True:
            smallest = pos
            left = 2 * pos + 1
            right = 2 * pos + 2

            if left < n and self._heap[left][0] < self._heap[smallest][0]:
                smallest = left
            if right < n and self._heap[right][0] < self._heap[smallest][0]:
                smallest = right

            if smallest != pos:
                self._swap(pos, smallest)
                pos = smallest
            else:
                break
