"""Advanced cache-eviction baselines for the master's-level comparison.

Implements three modern algorithms that the flagship GDSF policy must be
compared against:

* :class:`WTinyLFUPolicy` — Window-TinyLFU (Einziger, Manes, Friedman 2015/2017).
  Small LRU admission window + main SLRU with count-min-sketch admission.
* :class:`ARCPolicy` — Adaptive Replacement Cache (Megiddo & Modha, FAST 2003).
  Four-list (T1, T2, B1, B2) design with adaptive parameter ``p``.
* :class:`S3FIFOPolicy` — S3-FIFO (Yang et al., SOSP 2024). Small/Main/Ghost
  FIFO queues with a 2-bit frequency counter.

All three follow the same :class:`benchmarks.policies.CachePolicy` contract
(``put``, ``access``, ``reset``, ``name``, ``current_size``, ``max_size``)
so they can be plugged into the existing benchmark harness and replay driver.

Byte-based capacity is respected exactly; ghost/metadata structures are kept
outside the byte budget (they are cache-management overhead, not cached data).

Author: Nissim Brami
"""

from __future__ import annotations

import math
import random
from collections import OrderedDict, deque
from typing import Deque, Dict, List, Optional, Tuple

from benchmarks.policies import CachePolicy


# =============================================================================
# W-TinyLFU
# =============================================================================


class _CountMinSketch:
    """4-hash Count-Min Sketch with 4-bit saturating counters and aging."""

    _MAX = 15  # 4-bit saturating counter

    def __init__(self, width: int, seed: int = 0xC0FFEE) -> None:
        self.width = max(16, width)
        self._rows: List[List[int]] = [[0] * self.width for _ in range(4)]
        rng = random.Random(seed)
        # Independent hash seeds; kept small so ``hash`` mixing dominates.
        self._salts = [rng.randint(1, 2**31 - 1) for _ in range(4)]

    def _idx(self, row: int, key: str) -> int:
        return (hash((self._salts[row], key)) & 0x7FFFFFFF) % self.width

    def increment(self, key: str) -> None:
        for r in range(4):
            i = self._idx(r, key)
            if self._rows[r][i] < self._MAX:
                self._rows[r][i] += 1

    def estimate(self, key: str) -> int:
        return min(self._rows[r][self._idx(r, key)] for r in range(4))

    def age(self) -> None:
        """Halve every counter (right-shift). Called every W accesses."""
        for r in range(4):
            row = self._rows[r]
            for i in range(self.width):
                row[i] >>= 1

    def clear(self) -> None:
        for r in range(4):
            for i in range(self.width):
                self._rows[r][i] = 0


class _Doorkeeper:
    """Single-hash Bloom filter, cleared on each aging cycle."""

    def __init__(self, size: int, seed: int = 0xDEADBEEF) -> None:
        self.size = max(64, size)
        self._bits = bytearray((self.size + 7) // 8)
        self._salt = seed

    def _idx(self, key: str) -> int:
        return (hash((self._salt, key)) & 0x7FFFFFFF) % self.size

    def contains(self, key: str) -> bool:
        i = self._idx(key)
        return bool(self._bits[i >> 3] & (1 << (i & 7)))

    def add(self, key: str) -> None:
        i = self._idx(key)
        self._bits[i >> 3] |= 1 << (i & 7)

    def clear(self) -> None:
        for i in range(len(self._bits)):
            self._bits[i] = 0


class WTinyLFUPolicy(CachePolicy):
    """Window-TinyLFU (Einziger, Manes, Friedman, ACM TWEB 2017).

    Structure
    ---------
    - **Window LRU** (1% of byte capacity) admits every new item.
    - **Main SLRU** (99%) split 20% probation / 80% protected.
    - Candidate evicted from Window competes against Main-probation LRU tail
      via count-min-sketch estimated frequency; higher wins.
    - Sketch ages (halves counters) every ``W = 10 * estimated_capacity_items``
      accesses; doorkeeper cleared on the same cadence.

    Byte accounting is exact: bytes are attributed to the segment the item
    currently resides in, and each segment enforces its own byte budget by
    evicting the LRU tail until it fits.
    """

    def __init__(
        self,
        max_size: int,
        window_frac: float = 0.01,
        protected_frac: float = 0.80,
        seed: int = 0xC0FFEE,
    ) -> None:
        super().__init__(max_size)
        self._window_budget = max(1, int(max_size * window_frac))
        self._main_budget = max_size - self._window_budget
        self._protected_budget = max(1, int(self._main_budget * protected_frac))
        self._probation_budget = self._main_budget - self._protected_budget

        self._window: OrderedDict[str, Tuple[int, float]] = OrderedDict()
        self._probation: OrderedDict[str, Tuple[int, float]] = OrderedDict()
        self._protected: OrderedDict[str, Tuple[int, float]] = OrderedDict()
        self._win_bytes = 0
        self._prob_bytes = 0
        self._prot_bytes = 0

        # Sketch sized ~ 4 x expected_items; expected items ~ max_size / 512 B
        est_items = max(64, max_size // 512)
        self._sketch = _CountMinSketch(width=4 * est_items, seed=seed)
        self._doorkeeper = _Doorkeeper(size=est_items, seed=seed ^ 0x1)
        self._age_period = max(64, 10 * est_items)
        self._accesses_since_age = 0

    # -------- helpers ------------------------------------------------------

    def _touch_freq(self, key: str) -> None:
        if self._doorkeeper.contains(key):
            self._sketch.increment(key)
        else:
            self._doorkeeper.add(key)
        self._accesses_since_age += 1
        if self._accesses_since_age >= self._age_period:
            self._sketch.age()
            self._doorkeeper.clear()
            self._accesses_since_age = 0

    def _freq(self, key: str) -> int:
        # Doorkeeper contributes 1 bit; sketch supplies the rest.
        return (1 if self._doorkeeper.contains(key) else 0) + self._sketch.estimate(key)

    def _sum_bytes(self) -> int:
        return self._win_bytes + self._prob_bytes + self._prot_bytes

    def _evict_lru(self, od: OrderedDict, bytes_field: str) -> List[str]:
        """Pop the LRU tail (front) of ``od`` and update byte counter."""
        evicted: List[str] = []
        if not od:
            return evicted
        k, (sz, _) = od.popitem(last=False)
        setattr(self, bytes_field, getattr(self, bytes_field) - sz)
        evicted.append(k)
        return evicted

    def _admit_to_probation(self, key: str, size: int, cost: float) -> List[str]:
        """Insert ``key`` at probation MRU, evict LRU probation until it fits."""
        evicted: List[str] = []
        while self._prob_bytes + size > self._probation_budget and self._probation:
            evicted += self._evict_lru(self._probation, "_prob_bytes")
        if size <= self._probation_budget:
            self._probation[key] = (size, cost)
            self._probation.move_to_end(key)
            self._prob_bytes += size
        return evicted

    # -------- CachePolicy interface ---------------------------------------

    def put(self, key: str, size: int, cost: float) -> List[str]:
        if size > self.max_size:
            return []
        # Remove any existing copy first (put may act as an update)
        for seg, bytes_field in (
            (self._window, "_win_bytes"),
            (self._probation, "_prob_bytes"),
            (self._protected, "_prot_bytes"),
        ):
            if key in seg:
                old_size, _ = seg.pop(key)
                setattr(self, bytes_field, getattr(self, bytes_field) - old_size)

        self._touch_freq(key)

        evicted: List[str] = []
        # Insert into window MRU
        self._window[key] = (size, cost)
        self._window.move_to_end(key)
        self._win_bytes += size

        # If window overflows, pop candidate and let it fight for main entry
        while self._win_bytes > self._window_budget and self._window:
            cand_key, (cand_size, cand_cost) = self._window.popitem(last=False)
            self._win_bytes -= cand_size

            # Compare candidate vs probation LRU tail using sketch
            if self._probation and self._prob_bytes + cand_size > self._probation_budget:
                victim_key = next(iter(self._probation))  # LRU tail
                if self._freq(cand_key) > self._freq(victim_key):
                    evicted += self._admit_to_probation(cand_key, cand_size, cand_cost)
                else:
                    evicted.append(cand_key)  # candidate loses, discarded
            else:
                evicted += self._admit_to_probation(cand_key, cand_size, cand_cost)

        self.current_size = self._sum_bytes()
        return evicted

    def access(self, key: str) -> bool:
        self._touch_freq(key)
        if key in self._protected:
            self._protected.move_to_end(key)
            return True
        if key in self._probation:
            size, cost = self._probation.pop(key)
            self._prob_bytes -= size
            # Promote to protected MRU, demote protected LRU if overflow
            self._protected[key] = (size, cost)
            self._protected.move_to_end(key)
            self._prot_bytes += size
            while self._prot_bytes > self._protected_budget and self._protected:
                dk, (dsz, dcost) = self._protected.popitem(last=False)
                self._prot_bytes -= dsz
                # Demote back to probation MRU
                self._probation[dk] = (dsz, dcost)
                self._probation.move_to_end(dk)
                self._prob_bytes += dsz
                while self._prob_bytes > self._probation_budget and self._probation:
                    self._evict_lru(self._probation, "_prob_bytes")
            return True
        if key in self._window:
            self._window.move_to_end(key)
            return True
        return False

    def reset(self) -> None:
        self._window.clear()
        self._probation.clear()
        self._protected.clear()
        self._win_bytes = self._prob_bytes = self._prot_bytes = 0
        self._sketch.clear()
        self._doorkeeper.clear()
        self._accesses_since_age = 0
        self.current_size = 0

    @property
    def name(self) -> str:
        return "W-TinyLFU"


# =============================================================================
# ARC — Adaptive Replacement Cache
# =============================================================================


class ARCPolicy(CachePolicy):
    """Adaptive Replacement Cache (Megiddo & Modha, FAST 2003).

    Maintains four ordered structures over keys:

    * ``T1``  recent one-hit keys currently in cache (with data)
    * ``T2``  frequent (>=2 hits) keys currently in cache (with data)
    * ``B1``  ghost list of keys recently evicted from ``T1`` (no data)
    * ``B2``  ghost list of keys recently evicted from ``T2`` (no data)

    Adaptive parameter ``p`` (in bytes) shifts capacity between recency-favoring
    T1 and frequency-favoring T2 based on which ghost list gets hits.

    Byte accounting: T1 + T2 must fit in ``max_size`` bytes; ghost lists carry
    keys only and count toward metadata (not the byte budget).
    """

    def __init__(self, max_size: int) -> None:
        super().__init__(max_size)
        self._t1: OrderedDict[str, Tuple[int, float]] = OrderedDict()
        self._t2: OrderedDict[str, Tuple[int, float]] = OrderedDict()
        self._b1: OrderedDict[str, int] = OrderedDict()  # key -> size (remembered)
        self._b2: OrderedDict[str, int] = OrderedDict()
        self._t1_bytes = 0
        self._t2_bytes = 0
        self._p = 0  # adaptive target for |T1| in bytes

    def _replace(self, key_in_b2: bool) -> List[str]:
        """Evict from T1 or T2 based on adaptive parameter p."""
        evicted: List[str] = []
        if self._t1 and (self._t1_bytes > self._p or (key_in_b2 and self._t1_bytes == self._p)):
            if self._t1:
                k, (sz, _) = self._t1.popitem(last=False)
                self._t1_bytes -= sz
                self._b1[k] = sz
                self._b1.move_to_end(k)
                evicted.append(k)
        else:
            if self._t2:
                k, (sz, _) = self._t2.popitem(last=False)
                self._t2_bytes -= sz
                self._b2[k] = sz
                self._b2.move_to_end(k)
                evicted.append(k)
        return evicted

    def _trim_ghosts(self) -> None:
        # |B1| + |T1| <= c and total ghost <= c on byte terms is the classical
        # invariant; enforce via count in items (paper's original is item-based).
        cap = self.max_size
        while self._b1 and sum(self._b1.values()) + self._t1_bytes > cap:
            self._b1.popitem(last=False)
        while self._b2 and sum(self._b2.values()) + self._t2_bytes > 2 * cap:
            self._b2.popitem(last=False)

    def put(self, key: str, size: int, cost: float) -> List[str]:
        if size > self.max_size:
            return []
        evicted: List[str] = []

        if key in self._t1 or key in self._t2:
            # Update in place — move to T2 MRU
            if key in self._t1:
                old_size, _ = self._t1.pop(key)
                self._t1_bytes -= old_size
            else:
                old_size, _ = self._t2.pop(key)
                self._t2_bytes -= old_size
            self._t2[key] = (size, cost)
            self._t2.move_to_end(key)
            self._t2_bytes += size
        elif key in self._b1:
            # Ghost hit in B1: favor recency
            ghost_size = self._b1.pop(key)
            delta = max(len(self._b2) // max(1, len(self._b1) + 1), 1) if self._b1 else 1
            self._p = min(self._p + delta * max(1, size), self.max_size)
            while self._t1_bytes + self._t2_bytes + size > self.max_size:
                evicted += self._replace(key_in_b2=False)
                if not evicted:
                    break
            self._t2[key] = (size, cost)
            self._t2.move_to_end(key)
            self._t2_bytes += size
        elif key in self._b2:
            # Ghost hit in B2: favor frequency
            ghost_size = self._b2.pop(key)
            delta = max(len(self._b1) // max(1, len(self._b2) + 1), 1) if self._b2 else 1
            self._p = max(self._p - delta * max(1, size), 0)
            while self._t1_bytes + self._t2_bytes + size > self.max_size:
                evicted += self._replace(key_in_b2=True)
                if not evicted:
                    break
            self._t2[key] = (size, cost)
            self._t2.move_to_end(key)
            self._t2_bytes += size
        else:
            # Fresh miss
            while self._t1_bytes + self._t2_bytes + size > self.max_size:
                evicted += self._replace(key_in_b2=False)
                if not evicted:
                    break
            self._t1[key] = (size, cost)
            self._t1.move_to_end(key)
            self._t1_bytes += size

        self._trim_ghosts()
        self.current_size = self._t1_bytes + self._t2_bytes
        return evicted

    def access(self, key: str) -> bool:
        # Case I: key in T1 -> promote to T2
        if key in self._t1:
            size, cost = self._t1.pop(key)
            self._t1_bytes -= size
            self._t2[key] = (size, cost)
            self._t2.move_to_end(key)
            self._t2_bytes += size
            self.current_size = self._t1_bytes + self._t2_bytes
            return True
        if key in self._t2:
            self._t2.move_to_end(key)
            return True
        return False

    def reset(self) -> None:
        self._t1.clear()
        self._t2.clear()
        self._b1.clear()
        self._b2.clear()
        self._t1_bytes = self._t2_bytes = 0
        self._p = 0
        self.current_size = 0

    @property
    def name(self) -> str:
        return "ARC"


# =============================================================================
# S3-FIFO
# =============================================================================


class S3FIFOPolicy(CachePolicy):
    """S3-FIFO (Yang, Qiu, Yue, Berger — SOSP 2024).

    Three FIFO queues:

    * **S** (Small, 10% of byte capacity)
    * **M** (Main, 90% of byte capacity)
    * **G** (Ghost, key-only, size-bounded)

    Per-entry 2-bit saturating frequency counter incremented on hit.

    On miss:
      - If ``key`` in ``G``  -> insert into ``M`` (freq=0)
      - Otherwise             -> insert into ``S`` (freq=0)

    Eviction from ``S`` (when full):
      - if freq > 1 -> promote to ``M``
      - else        -> discard, push key to ``G``

    Eviction from ``M`` (when full):
      - if freq > 0 -> decrement freq, reinsert at tail
      - else        -> discard
    """

    def __init__(self, max_size: int, small_frac: float = 0.10) -> None:
        super().__init__(max_size)
        self._small_budget = max(1, int(max_size * small_frac))
        self._main_budget = max_size - self._small_budget

        # OrderedDict preserves FIFO order (front = oldest)
        self._s: OrderedDict[str, Tuple[int, float]] = OrderedDict()
        self._m: OrderedDict[str, Tuple[int, float]] = OrderedDict()
        self._freq: Dict[str, int] = {}
        self._s_bytes = 0
        self._m_bytes = 0

        # Ghost bounded to ~size of Main in *number of keys*, FIFO
        self._ghost: Deque[str] = deque()
        self._ghost_set: set = set()
        self._ghost_cap = max(64, self._main_budget // 512)

    # -------- helpers ------------------------------------------------------

    def _ghost_add(self, key: str) -> None:
        if key in self._ghost_set:
            return
        self._ghost.append(key)
        self._ghost_set.add(key)
        while len(self._ghost) > self._ghost_cap:
            evicted = self._ghost.popleft()
            self._ghost_set.discard(evicted)

    def _evict_from_s(self) -> List[str]:
        """Pop head of S; promote to M if hot, else ghost it."""
        evicted: List[str] = []
        if not self._s:
            return evicted
        k, (sz, cost) = self._s.popitem(last=False)
        self._s_bytes -= sz
        f = self._freq.pop(k, 0)
        if f > 1:
            # Hot: move to M (reset freq to 0 per paper)
            self._m[k] = (sz, cost)
            self._m_bytes += sz
            self._freq[k] = 0
            # M may now overflow — drain
            while self._m_bytes > self._main_budget and self._m:
                evicted += self._evict_from_m()
        else:
            evicted.append(k)
            self._ghost_add(k)
        return evicted

    def _evict_from_m(self) -> List[str]:
        """Pop head of M; requeue if warm, else discard."""
        evicted: List[str] = []
        # Bounded scan to avoid infinite loop on all-warm state
        rotations = 0
        while self._m and rotations < len(self._m) + 1:
            k, (sz, cost) = self._m.popitem(last=False)
            self._m_bytes -= sz
            f = self._freq.get(k, 0)
            if f > 0:
                self._freq[k] = f - 1
                self._m[k] = (sz, cost)
                self._m_bytes += sz
                rotations += 1
            else:
                self._freq.pop(k, None)
                evicted.append(k)
                return evicted
        # All entries were warm and got demoted — evict the front now
        if self._m and not evicted:
            k, (sz, cost) = self._m.popitem(last=False)
            self._m_bytes -= sz
            self._freq.pop(k, None)
            evicted.append(k)
        return evicted

    # -------- CachePolicy interface ---------------------------------------

    def put(self, key: str, size: int, cost: float) -> List[str]:
        if size > self.max_size:
            return []
        evicted: List[str] = []

        # Update-in-place: if already resident, remove first
        if key in self._s:
            old_sz, _ = self._s.pop(key)
            self._s_bytes -= old_sz
            self._freq.pop(key, None)
        elif key in self._m:
            old_sz, _ = self._m.pop(key)
            self._m_bytes -= old_sz
            self._freq.pop(key, None)

        if key in self._ghost_set:
            # Ghost hit: admit to M
            self._ghost_set.discard(key)
            try:
                self._ghost.remove(key)
            except ValueError:
                pass
            while self._m_bytes + size > self._main_budget and self._m:
                evicted += self._evict_from_m()
            self._m[key] = (size, cost)
            self._m_bytes += size
            self._freq[key] = 0
        else:
            # Fresh miss: admit to S
            while self._s_bytes + size > self._small_budget and self._s:
                evicted += self._evict_from_s()
            self._s[key] = (size, cost)
            self._s_bytes += size
            self._freq[key] = 0

        self.current_size = self._s_bytes + self._m_bytes
        return evicted

    def access(self, key: str) -> bool:
        if key in self._s or key in self._m:
            self._freq[key] = min(3, self._freq.get(key, 0) + 1)
            return True
        return False

    def reset(self) -> None:
        self._s.clear()
        self._m.clear()
        self._freq.clear()
        self._ghost.clear()
        self._ghost_set.clear()
        self._s_bytes = self._m_bytes = 0
        self.current_size = 0

    @property
    def name(self) -> str:
        return "S3-FIFO"


# =============================================================================
# Registration hook
# =============================================================================


def register_advanced_policies() -> None:
    """Register the three advanced policies into POLICY_REGISTRY.

    Idempotent; safe to call multiple times.
    """
    from benchmarks import policies as _p
    _p.POLICY_REGISTRY.setdefault("W-TinyLFU", WTinyLFUPolicy)
    _p.POLICY_REGISTRY.setdefault("ARC", ARCPolicy)
    _p.POLICY_REGISTRY.setdefault("S3-FIFO", S3FIFOPolicy)


register_advanced_policies()
