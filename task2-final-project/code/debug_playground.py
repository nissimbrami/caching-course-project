"""
Interactive debug playground for the GDSF cost-aware eviction plugin.

How to use
----------
1. Terminal:  python debug_playground.py
   Runs all sections and prints step-by-step output.

2. Interactive:  python -i debug_playground.py
   Runs everything then drops you into a REPL with `m`, `plugin`, `lru_res`,
   `gdsf_res` etc. still in scope. Type e.g. `m.get_priority("A")`.

3. VS Code debugger:
   Set a breakpoint on any line, press F5 -> "Python File".
   F10 = step over, F11 = step into, Shift+F11 = step out.
   Useful step-into targets:
     - m.put(...)         -> src/cost_aware_eviction/eviction_manager.py:put
     - m._compute_priority(...) -> the GDSF formula itself
     - m._evict_one()     -> the eviction decision
     - run_single_experiment(...) -> benchmarks/runner.py

Nothing in this file changes any project state; it only reads/exercises the
public API. Safe to delete.
"""
from src.cost_aware_eviction.eviction_manager import GDSFEvictionManager
from src.cost_aware_eviction.priority_queue import IndexedMinHeap
from src.cost_aware_eviction.gptcache_plugin import GDSFEvictionPlugin
from benchmarks.workloads import generate_high_variance_cost_workload
from benchmarks.runner import run_single_experiment, create_policy


def section(title):
    print()
    print("=" * 72)
    print("  " + title)
    print("=" * 72)


# ---------------------------------------------------------------------------
section("1. IndexedMinHeap primitive")
# ---------------------------------------------------------------------------
h = IndexedMinHeap()
for k, p in [("a", 3.0), ("b", 1.0), ("c", 2.0), ("d", 5.0), ("e", 0.5)]:
    h.push(k, p)
    print(f"  push({k!r}, {p})  size={len(h)}  min={h.peek()}")
print(f"  update('a', 0.1)  # promotes 'a' from priority 3.0 to 0.1")
h.update("a", 0.1)
print("  Pop order (lowest priority first):")
while len(h):
    print(f"    pop -> {h.pop()}")


# ---------------------------------------------------------------------------
section("2. GDSFEvictionManager: watch the priority calculation")
# ---------------------------------------------------------------------------
m = GDSFEvictionManager(max_size=100, alpha=1.0, beta=1.0)
print("  cache cap = 100 bytes, alpha=1.0, beta=1.0")
print("  Priority(i) = Clock + freq(i)^alpha * cost(i)^beta / size(i)")
print()

m.put("cheap",     size=40, cost=0.5)   # low cost
m.put("expensive", size=30, cost=2.0)   # high cost
m.put("small",     size=20, cost=0.1)   # tiny cost
print(f"  After 3 puts:   size = {m.current_size} / {m.max_size}")
print(f"  clock (L) = {m.clock:.4f}")
for key in ("cheap", "expensive", "small"):
    print(f"  priority({key:9s}) = {m.get_priority(key):.4f}")

print()
print("  Now access 'expensive' twice, 'cheap' once (build up freq):")
m.access("expensive")
m.access("expensive")
m.access("cheap")
for key in ("cheap", "expensive", "small"):
    print(f"  priority({key:9s}) = {m.get_priority(key):.4f}")

print()
print("  Insert a 50-byte item -> forces eviction (need >= 20 bytes free):")
m.put("newcomer", size=50, cost=1.5)
for key in ("cheap", "expensive", "small", "newcomer"):
    tag = "IN " if key in m else "OUT"
    print(f"    {tag} {key}")


# ---------------------------------------------------------------------------
section("3. GPTCache plugin -- subclass of the article's EvictionBase")
# ---------------------------------------------------------------------------
try:
    from gptcache.manager.eviction.base import EvictionBase
    real_gptcache = True
except ImportError:
    real_gptcache = False

plugin = GDSFEvictionPlugin(maxsize=200)
print(f"  gptcache library available at runtime? {real_gptcache}")
if real_gptcache:
    print(f"  isinstance(plugin, gptcache.EvictionBase) = {isinstance(plugin, EvictionBase)}")
print(f"  plugin.policy = {plugin.policy!r}")

plugin.register_metadata("k1", size=50, cost=1.0)
plugin.register_metadata("k2", size=80, cost=3.0)
plugin.put(["k1", "k2"])
print(f"  after putting k1(50B, $1) + k2(80B, $3):  num_entries={plugin.num_entries}, current_size={plugin.current_size}")

plugin.register_metadata("k3", size=120, cost=0.5)
plugin.put(["k3"])   # forces eviction: 50 + 80 + 120 = 250 > 200
print(f"  after putting k3(120B, $0.5) -> over cap:  num_entries={plugin.num_entries}, current_size={plugin.current_size}")


# ---------------------------------------------------------------------------
section("4. Mini benchmark: LRU vs GDSF on high-variance-cost workload")
# ---------------------------------------------------------------------------
wl = generate_high_variance_cost_workload(n_queries=500, n_unique=80, seed=42)
cache_size = 10_000
results = {}
for name in ("LRU", "LFU", "GDSF"):
    r = run_single_experiment(
        policy=create_policy(name, cache_size),
        workload=wl,
        workload_name="high_variance_cost",
        cache_size=cache_size,
        run_id=0,
        seed=42,
    )
    results[name] = r
    print(f"  {name:5s}  hit_rate={r.hit_rate:.3f}  cwhr={r.cost_weighted_hit_rate:.3f}  savings=${r.dollar_savings:.2f}")

lru = results["LRU"]
gdsf = results["GDSF"]
delta_cwhr = gdsf.cost_weighted_hit_rate - lru.cost_weighted_hit_rate
delta_savings = gdsf.dollar_savings - lru.dollar_savings
print()
print(f"  GDSF vs LRU:  Delta CWHR = {delta_cwhr:+.3f}, extra savings = ${delta_savings:+.2f}")
print("  (This is exactly the paper's thesis running live in your terminal.)")


# ---------------------------------------------------------------------------
section("Playground ready")
# ---------------------------------------------------------------------------
print("  Live objects available if you ran with `python -i debug_playground.py`:")
print("    m       -> GDSFEvictionManager instance from section 2")
print("    plugin  -> GDSFEvictionPlugin instance from section 3")
print("    results -> dict of BenchmarkResult from section 4")
print()
print("  Try:")
print("    >>> m.get_priority('expensive')")
print("    >>> plugin.next_victim()")
print("    >>> results['GDSF'].__dict__")
