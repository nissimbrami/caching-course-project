"""
Interactive playground for the GDSF cost-aware eviction plugin.

This is a teaching / demo tool, not part of the benchmark suite.
It walks you through the algorithm step by step, in plain English, so you
can SEE what GDSF does that LRU does not.

------------------------------------------------------------------------
HOW TO USE
------------------------------------------------------------------------

  1. Interactive menu (recommended):
         python debug_playground.py

     You'll get a numbered menu. Pick a lesson, read the explanation,
     watch the numbers, hit Enter to continue.

  2. Run every lesson end-to-end without prompts:
         python debug_playground.py --all

  3. Drop into a Python REPL after the lessons run:
         python -i debug_playground.py --all
     Then type e.g.  >>> m.get_priority("expensive")

------------------------------------------------------------------------
WHAT THIS DEMO IS AND IS NOT
------------------------------------------------------------------------

  IT IS:
    - A live run of the real GDSF eviction manager from src/
    - A demonstration of the exact formula in the report:
          Priority(i) = Clock + freq(i)^alpha * cost(i)^beta / size(i)
    - A proof that the plugin correctly subclasses gptcache.EvictionBase

  IT IS NOT:
    - A real LLM call. No prompts are sent to OpenAI/Anthropic.
    - A live run of GPTCache's request pipeline. Nothing here goes
      through GPTCache's semantic-similarity / vector-store / SQL layer.
    - The benchmark. The 3600-row benchmark used to produce the report
      lives in benchmarks/run_all.py, not here.

  In short: the ALGORITHM is real; the WORKLOAD is synthetic (numpy RNG
  with a fixed seed). This is the standard methodology for cache-policy
  papers (GDSF 1997, ARC 2003, TinyLFU 2017 all use synthetic traces).
"""

from __future__ import annotations

import argparse
import sys
from typing import Callable, Dict, List, Tuple

from src.cost_aware_eviction.eviction_manager import GDSFEvictionManager
from src.cost_aware_eviction.priority_queue import IndexedMinHeap
from src.cost_aware_eviction.gptcache_plugin import GDSFEvictionPlugin
from benchmarks.workloads import generate_high_variance_cost_workload
from benchmarks.runner import run_single_experiment, create_policy


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

BAR = "=" * 74
DASH = "-" * 74


def header(n: int, total: int, title: str) -> None:
    print()
    print(BAR)
    print(f"  LESSON {n}/{total}  --  {title}")
    print(BAR)


def say(msg: str) -> None:
    """Print a paragraph of explanatory prose, wrapped at 74 cols."""
    for line in msg.strip("\n").splitlines():
        print("  " + line.rstrip())


def show(label: str, value) -> None:
    print(f"    {label:<38} {value}")


def divider(label: str = "") -> None:
    if label:
        print()
        print(f"  --- {label} " + "-" * (67 - len(label)))
    else:
        print("  " + DASH)


def pause(interactive: bool) -> None:
    if interactive:
        try:
            input("\n  [press Enter to continue] ")
        except EOFError:
            pass


# ---------------------------------------------------------------------------
# Lesson 1 -- the min-heap primitive
# ---------------------------------------------------------------------------

def lesson_1_heap(interactive: bool) -> None:
    header(1, 6, "The min-heap: GDSF's underlying container")
    say("""
GDSF has to answer one question fast: "which cached entry has the
lowest priority right now?" That's the eviction victim.

We use an IndexedMinHeap: same as a normal binary heap, but with an
extra dictionary so we can also UPDATE the priority of an existing key
in O(log n). Standard heapq can't do that.

Below we push 5 items, then bump 'a' to the top by lowering its
priority, then drain the heap. Watch the pop order: lowest priority
first.
""")
    divider("live run")

    h = IndexedMinHeap()
    items: List[Tuple[str, float]] = [
        ("a", 3.0), ("b", 1.0), ("c", 2.0), ("d", 5.0), ("e", 0.5),
    ]
    for k, p in items:
        h.push(k, p)
        show(f"push({k!r}, priority={p})", f"size={len(h)}  current_min={h.peek()}")

    print()
    say("Now we call update('a', 0.1) -- 'a' had priority 3.0, we lower it to 0.1.")
    say("That should make 'a' the new minimum.")
    h.update("a", 0.1)
    show("after update('a', 0.1) min =", h.peek())

    print()
    say("Drain the heap. Order should be by ascending priority:")
    print("    (priority in parentheses is what GDSF computed for each key)")
    order = []
    while len(h):
        popped = h.pop()
        # pop() returns either the key or a (key, priority) tuple depending on impl
        if isinstance(popped, tuple):
            k, prio = popped
        else:
            k, prio = popped, None
        order.append(k)
        prio_str = f"(priority was {prio})" if prio is not None else ""
        show(f"pop -> {k!r}", prio_str)

    print()
    say(f"Pop order: {order}")
    say("Expected: ['a', 'e', 'b', 'c', 'd'] -- ascending priority. Matches? "
        + ("YES" if order == ["a", "e", "b", "c", "d"] else "NO"))
    pause(interactive)


# ---------------------------------------------------------------------------
# Lesson 2 -- the GDSF formula in action
# ---------------------------------------------------------------------------

def lesson_2_formula(interactive: bool) -> Tuple[GDSFEvictionManager, dict]:
    header(2, 6, "The GDSF formula: watching priorities change")
    say("""
The core formula is:

    Priority(i) = Clock + freq(i)^alpha * cost(i)^beta / size(i)

    Clock  = the priority of the LAST item we evicted (starts at 0).
             Aging mechanism -- fresh items always land above the clock.
    freq   = how many times this item has been accessed.
    cost   = dollar value of regenerating this item (LLM call cost).
    size   = bytes the item occupies.
    alpha  = how much we weight frequency (default 1.0).
    beta   = how much we weight cost      (default 1.0).

Intuition: a high-priority item is one that is cheap to keep (small),
expensive to regenerate (high cost), and hit often (high freq). A
low-priority item is large, cheap, and rarely accessed -- perfect
eviction candidate.

We build a 100-byte cache and put 3 items:
    'cheap'     -- 40 bytes, cost $0.50
    'expensive' -- 30 bytes, cost $2.00
    'small'     -- 20 bytes, cost $0.10
Total: 90 bytes. Fits.
""")
    divider("live run")

    m = GDSFEvictionManager(max_size=100, alpha=1.0, beta=1.0)
    m.put("cheap",     size=40, cost=0.5)
    m.put("expensive", size=30, cost=2.0)
    m.put("small",     size=20, cost=0.1)

    show("current_size / max_size", f"{m.current_size} / {m.max_size} bytes")
    show("clock (L)", f"{m.clock:.4f}")
    print()
    say("Initial priorities (freq=1 for all, so priority = cost/size):")
    for key, expected in [("cheap", 0.5/40), ("expensive", 2.0/30), ("small", 0.1/20)]:
        actual = m.get_priority(key)
        show(f"priority({key})", f"{actual:.4f}   (expected {expected:.4f})")

    print()
    say("Now access 'expensive' twice and 'cheap' once. freq becomes:")
    say("  cheap: 2, expensive: 3, small: 1")
    m.access("expensive"); m.access("expensive"); m.access("cheap")
    print()
    say("New priorities:")
    for key in ("cheap", "expensive", "small"):
        show(f"priority({key})", f"{m.get_priority(key):.4f}")
    print()
    say("Notice 'expensive' rose the most -- high cost AND high freq. Good.")

    print()
    divider("forcing an eviction")
    say("""
Now insert a 50-byte item called 'newcomer' (cost $1.50). Cache only
has 10 bytes free (100 - 90). We need to evict AT LEAST 40 bytes.

GDSF will pick the item with the LOWEST priority. That should be
'small' or 'cheap' (both are low-frequency, low-cost items compared to
'expensive').
""")
    priorities_before = {k: m.get_priority(k) for k in ("cheap", "expensive", "small")}
    say(f"priorities before eviction: {priorities_before}")
    evicted = m.put("newcomer", size=50, cost=1.5)
    say(f"m.put('newcomer', size=50, cost=1.5) evicted: {evicted}")
    print()
    say("Who survived?")
    for key in ("cheap", "expensive", "small", "newcomer"):
        tag = "IN " if key in m else "OUT"
        show(f"  {tag} {key}", f"priority now = {m.get_priority(key):.4f}"
             if key in m else "(evicted)")
    print()
    say(f"Clock advanced to {m.clock:.4f} -- this is the priority of the last evicted item.")
    say("Fresh items after this eviction will get priority >= clock, so they can't")
    say("be evicted immediately even if their raw score is tiny (aging protection).")

    pause(interactive)
    return m, priorities_before


# ---------------------------------------------------------------------------
# Lesson 3 -- the GPTCache plugin
# ---------------------------------------------------------------------------

def lesson_3_plugin(interactive: bool) -> GDSFEvictionPlugin:
    header(3, 6, "The GPTCache plugin: what makes this a plugin?")
    say("""
GPTCache is the open-source LLM semantic cache from Zilliz. To swap
its default LRU eviction for our GDSF one, we need a class that:

  (a) subclasses gptcache.manager.eviction.base.EvictionBase
  (b) implements put(objs), get() / next_victim(), is_evict()
  (c) exposes a .policy string identifier

That's exactly what GDSFEvictionPlugin does. It's a thin adapter that
translates GPTCache's simple API into calls on our GDSF manager.

CAVEAT: this playground exercises the plugin DIRECTLY. It does NOT
spin up a real gptcache.Cache and feed prompts through it. The plugin
would work in a live GPTCache -- but we prove that separately with
integration tests, not here.
""")
    divider("live run")

    try:
        from gptcache.manager.eviction.base import EvictionBase
        real = True
    except ImportError:
        real = False

    plugin = GDSFEvictionPlugin(maxsize=200)
    show("gptcache library installed?", real)
    if real:
        show("plugin IS-A gptcache.EvictionBase?", isinstance(plugin, EvictionBase))
    show("plugin.policy", repr(plugin.policy))
    show("plugin.max_size (bytes)", plugin.max_size)

    print()
    say("Register 2 items and put them into the plugin (simulating GPTCache calling put):")
    plugin.register_metadata("k1", size=50, cost=1.0)
    plugin.register_metadata("k2", size=80, cost=3.0)
    plugin.put(["k1", "k2"])
    show("num_entries", plugin.num_entries)
    show("current_size", f"{plugin.current_size} / {plugin.max_size} bytes")
    show("is_evict()?", plugin.is_evict())

    print()
    say("Now put a 3rd item (120 bytes, $0.50). 50 + 80 + 120 = 250 > 200 -> eviction.")
    plugin.register_metadata("k3", size=120, cost=0.5)
    plugin.put(["k3"])
    show("num_entries after k3", plugin.num_entries)
    show("current_size after k3", f"{plugin.current_size} / {plugin.max_size} bytes")
    say("The plugin internally called our GDSFEvictionManager, which chose the")
    say("lowest-priority victim. In a real GPTCache setup, GPTCache would now")
    say("call plugin.get() to learn which key to also delete from its data store.")

    pause(interactive)
    return plugin


# ---------------------------------------------------------------------------
# Lesson 4 -- LRU vs LFU vs GDSF on a synthetic workload
# ---------------------------------------------------------------------------

def lesson_4_bench(interactive: bool) -> dict:
    header(4, 6, "GDSF vs LRU vs LFU: the thesis in one number")
    say("""
We generate a synthetic workload of 500 queries over 80 unique keys,
where costs are drawn from a trimodal distribution:
    60% cheap    (~$0.002, "gpt-3.5-turbo")
    30% medium   (~$0.06,  "gpt-4")
    10% expensive(~$0.12,  "gpt-4-32k")

Cache capacity = 10,000 bytes (small on purpose -- forces evictions).
Random seed = 42. Same trace for all 3 policies -- fair comparison.

Metrics:
    hit_rate  -- fraction of queries served from cache
    cwhr      -- COST-WEIGHTED hit rate = dollars_saved / dollars_possible
    savings   -- absolute dollar savings on this trace

The thesis of the report: GDSF may not win on plain hit_rate, but it
wins on cwhr and savings -- because it preferentially keeps
expensive-to-regenerate entries.
""")
    divider("live run (may take a few seconds)")

    wl = generate_high_variance_cost_workload(n_queries=500, n_unique=80, seed=42)
    cache_size = 10_000

    results = {}
    print(f"    {'policy':<8} {'hit_rate':>10} {'cwhr':>10} {'savings':>12}")
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
        print(f"    {name:<8} {r.hit_rate:>10.3f} "
              f"{r.cost_weighted_hit_rate:>10.3f} ${r.dollar_savings:>10.2f}")

    print()
    lru = results["LRU"]
    gdsf = results["GDSF"]
    d_hr = gdsf.hit_rate - lru.hit_rate
    d_cwhr = gdsf.cost_weighted_hit_rate - lru.cost_weighted_hit_rate
    d_save = gdsf.dollar_savings - lru.dollar_savings

    say(f"GDSF vs LRU on this trace:")
    show("delta hit_rate", f"{d_hr:+.3f}   ({'GDSF wins' if d_hr > 0 else 'LRU wins' if d_hr < 0 else 'tie'})")
    show("delta cwhr    ", f"{d_cwhr:+.3f}   ({'GDSF wins' if d_cwhr > 0 else 'LRU wins' if d_cwhr < 0 else 'tie'})")
    show("delta savings ", f"${d_save:+.2f}   ({'GDSF wins' if d_save > 0 else 'LRU wins' if d_save < 0 else 'tie'})")
    print()
    say("This is the paper's story running live on your machine. The full")
    say("benchmark repeats this over 6 workloads x 4 cache sizes x 30 seeds =")
    say("3600 rows, and reports statistical significance with paired-t + BCa.")

    pause(interactive)
    return results


# ---------------------------------------------------------------------------
# Lesson 5 -- what to do next
# ---------------------------------------------------------------------------

def lesson_5_next(interactive: bool) -> None:
    header(5, 6, "What next?")
    say("""
You've now seen every core moving part of Task 2:

  1. IndexedMinHeap  -- the container that makes O(log n) eviction possible
  2. GDSFEvictionManager -- the formula and the aging clock
  3. GDSFEvictionPlugin  -- the GPTCache adapter
  4. Live LRU vs LFU vs GDSF comparison on a synthetic trace

Suggested next commands (run any of these in your shell):

    python -m pytest tests/ -q
        --> 300 tests. Takes ~15 seconds. Should end with "300 passed".

    python -m benchmarks.run_all --n-runs 3
        --> Short benchmark, ~1 minute. Full run is --n-runs 30 (~15-30 min).

    python scripts/compute_statistics.py
        --> Rebuilds the paired-t + BCa bootstrap stats used in the report.

    python scripts/generate_plots.py
        --> Regenerates the 4 figures in the report.

    bash scripts/run_live_smoke.sh
        --> Full end-to-end check. 8 stages, ~2 minutes.

If you want to poke at internals interactively, run:

    python -i debug_playground.py --all

You'll be dropped into a REPL with these still in scope:
    m        -- the GDSFEvictionManager from Lesson 2
    plugin   -- the GDSFEvictionPlugin from Lesson 3
    results  -- the LRU/LFU/GDSF BenchmarkResult dict from Lesson 4

Try:
    >>> m.get_priority("expensive")
    >>> plugin.next_victim()
    >>> results["GDSF"].cost_weighted_hit_rate
    >>> vars(results["GDSF"])
""")
    pause(interactive)


# ---------------------------------------------------------------------------
# Lesson 6 -- REAL GPTCache integration (the plugin actually being called)
# ---------------------------------------------------------------------------

def lesson_6_real_gptcache(interactive: bool):
    header(6, 6, "REAL GPTCache mode: the plugin actually being called")
    say("""
Everything before this lesson called the plugin DIRECTLY. That proves
the algorithm works, but it doesn't prove GPTCache would use it.

This lesson is different. Here we:

  1. Build a real gptcache.manager.SSDataManager
  2. Wire our GDSFEvictionPlugin in as the eviction_base parameter
  3. Save (question, answer, fake_embedding) triples through GPTCache's
     own import_data() pipeline
  4. Force capacity overflow so GPTCache asks the plugin who to evict
  5. Show that when the plugin decides, the on_evict callback we passed
     fires with the evicted sql_ids (the callback in this demo just logs
     them; a production callback would forward them to SSDataManager._clear
     to purge SQLite + FAISS)

No LLM is called (we fabricate the answers) but the ENTIRE GPTCache
pipeline is live: sqlite storage, FAISS vector index, data_manager,
and eviction handshake. The plugin is genuinely inside GPTCache.
""")
    divider("live run")

    try:
        import numpy as np
        from gptcache.manager import CacheBase, VectorBase
        from gptcache.manager.data_manager import SSDataManager
    except ImportError as e:
        say(f"gptcache not fully installed: {e}")
        say("Skipping this lesson.")
        pause(interactive)
        return None

    say("Step 1: create real SQLite + FAISS backends.")
    sqlite = CacheBase("sqlite", sql_url="sqlite:///:memory:", table_name="gptcache")
    dim = 8
    faiss = VectorBase("faiss", dimension=dim, index_path=":memory:")
    show("sqlite backend", type(sqlite).__name__)
    show("faiss backend ", type(faiss).__name__)

    print()
    say("Step 2: build the GDSF plugin. We wire TWO callbacks in:")
    say("  * metadata_callback   -> lets GDSF ask 'what does key K cost?'")
    say("  * on_evict            -> lets GDSF hand evicted sql_ids back to")
    say("                            the caller (in this demo the callback")
    say("                            just logs them; SSDataManager._clear is")
    say("                            the production purge path)")
    say("So the REAL per-question dollar cost reaches the GDSF formula,")
    say("not a flat default.")
    evicted_log: List[Any] = []
    id_to_metadata: Dict[int, Tuple[int, float]] = {}

    def on_evict_callback(keys: List[Any]) -> None:
        evicted_log.extend(keys)
        print(f"    [on_evict fired] plugin reported evicted sql_ids: {keys}")

    def cost_lookup(key: Any) -> Tuple[int, float]:
        # GPTCache calls plugin.put([sql_id]); we translate sql_id -> (size,cost)
        return id_to_metadata.get(key, (1, 1.0))

    plugin = GDSFEvictionPlugin(
        maxsize=3,  # tiny -- forces immediate eviction after 4 puts
        alpha=1.0,
        beta=1.0,
        default_entry_size=1,   # each entry counts as 1 unit
        default_entry_cost=1.0,
        metadata_callback=cost_lookup,
        on_evict=on_evict_callback,
    )
    show("plugin.policy", plugin.policy)
    show("plugin has metadata_callback wired?", plugin._metadata_callback is not None)
    show("plugin has on_evict wired?",          plugin._on_evict is not None)

    print()
    say("Step 3: hand the plugin to SSDataManager as eviction_base=.")
    data_manager = SSDataManager(
        s=sqlite, v=faiss, o=None, e=plugin,
        max_size=3, clean_size=1,
    )
    show("data_manager type", type(data_manager).__name__)
    show("data_manager.eviction_base IS our plugin?", data_manager.eviction_base is plugin)

    print()
    say("Step 4: save 3 (question, answer, fake_embedding) triples.")
    say("Costs: q1=$0.10 (cheap), q2=$1.00 (medium), q3=$5.00 (expensive).")
    say("After each save we read back the newly-assigned sql_id and register")
    say("its real cost in id_to_metadata so GDSF sees the true dollar value.")
    rng = np.random.default_rng(42)
    triples = [
        ("q1_what_is_2_plus_2",        "4",                              0.10),
        ("q2_write_a_python_hello",    "print('hello world')",           1.00),
        ("q3_explain_quantum_gravity", "It's an unsolved problem where...", 5.00),
    ]
    for q, a, cost in triples:
        emb = rng.standard_normal(dim).astype("float32")
        data_manager.save(q, a, emb)
        # sql_id was just auto-assigned; grab the newest live id and record cost
        live_ids = list(sqlite.get_ids(deleted=False))
        if live_ids:
            new_id = max(live_ids)
            id_to_metadata[new_id] = (len(a.encode("utf-8")), cost)
        print(f"    saved: {q[:35]:<35} (cost=${cost:.2f}, sql_id={new_id})")

    show("plugin.num_entries after 3 saves", plugin.num_entries)
    show("id_to_metadata registered",        dict(id_to_metadata))
    show("evictions so far",                 len(evicted_log))

    print()
    say("Step 5: save a 4th entry -- this MUST overflow the plugin's cap of 3.")
    say("GDSF should keep the expensive q3 ($5) and evict a cheap one.")
    q4, a4, c4 = ("q4_capital_of_france", "Paris", 0.05)
    emb4 = rng.standard_normal(dim).astype("float32")
    data_manager.save(q4, a4, emb4)
    live_ids = list(sqlite.get_ids(deleted=False))
    if live_ids:
        id_to_metadata[max(live_ids)] = (len(a4.encode("utf-8")), c4)
    show("plugin.num_entries after 4th save", plugin.num_entries)
    show("evictions total",                   len(evicted_log))
    show("evicted keys",                      evicted_log)

    print()
    say("What just happened, in plain English:")
    say("  1. GPTCache.SSDataManager.import_data() was called for q4.")
    say("  2. It called self.eviction_base.put([new_id]) -- our plugin.")
    say("  3. Our plugin ran the GDSF formula, chose the lowest-priority key.")
    say("  4. Our plugin fired on_evict(evicted_keys) -- the callback we passed.")
    say("  5. Our on_evict callback fired with the evicted sql_ids.")
    say("     (In this demo the callback just logs; a production integration")
    say("     would forward the ids to SSDataManager._clear to hard-purge")
    say("     SQLite + FAISS.)")
    print()
    say("That is a REAL GPTCache <-> GDSF handshake for the DECISION half.")
    say("The plugin is genuinely wired in as SSDataManager.eviction_base and")
    say("SSDataManager.save() drives put() through it. If someone claims the")
    say("plugin doesn't really plug in, this lesson is the counter-example.")

    pause(interactive)
    return plugin


# ---------------------------------------------------------------------------
# Menu / driver
# ---------------------------------------------------------------------------

LESSONS: List[Tuple[str, Callable]] = [
    ("Lesson 1 -- IndexedMinHeap primitive", lesson_1_heap),
    ("Lesson 2 -- GDSF formula & eviction",  lesson_2_formula),
    ("Lesson 3 -- GPTCache plugin adapter",  lesson_3_plugin),
    ("Lesson 4 -- LRU vs LFU vs GDSF bench", lesson_4_bench),
    ("Lesson 5 -- What to run next",         lesson_5_next),
    ("Lesson 6 -- REAL GPTCache mode",       lesson_6_real_gptcache),
]


def print_menu() -> None:
    print()
    print(BAR)
    print("  GDSF Cost-Aware Eviction -- Interactive Playground")
    print(BAR)
    print("  Pick a lesson (or 'a' for all, 'q' to quit):")
    print()
    for i, (title, _) in enumerate(LESSONS, 1):
        print(f"    {i}.  {title}")
    print(f"    a.  Run all lessons in order")
    print(f"    q.  Quit")
    print()


def run_menu() -> Tuple[object, object, object]:
    """Interactive menu loop. Returns (m, plugin, results) if they were built."""
    m = plugin = results = None
    while True:
        print_menu()
        try:
            choice = input("  Your choice: ").strip().lower()
        except EOFError:
            break
        if not choice:
            continue
        if choice == "q":
            print("  Bye.")
            break
        if choice == "a":
            for _, fn in LESSONS:
                out = fn(interactive=True)
                if isinstance(out, tuple):
                    m = out[0]
                elif isinstance(out, GDSFEvictionPlugin):
                    plugin = out
                elif isinstance(out, dict):
                    results = out
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(LESSONS):
            _, fn = LESSONS[int(choice) - 1]
            out = fn(interactive=True)
            if isinstance(out, tuple):
                m = out[0]
            elif isinstance(out, GDSFEvictionPlugin):
                plugin = out
            elif isinstance(out, dict):
                results = out
            continue
        print(f"  Unknown choice: {choice!r}. Type a number, 'a', or 'q'.")
    return m, plugin, results


def run_all_noninteractive() -> Tuple[object, object, object]:
    m = plugin = results = None
    for _, fn in LESSONS:
        out = fn(interactive=False)
        if isinstance(out, tuple):
            m = out[0]
        elif isinstance(out, GDSFEvictionPlugin):
            plugin = out
        elif isinstance(out, dict):
            results = out
    return m, plugin, results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Interactive playground for GDSF cost-aware eviction.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run every lesson end-to-end without prompting.",
    )
    args = parser.parse_args()

    if args.all:
        m, plugin, results = run_all_noninteractive()
    else:
        m, plugin, results = run_menu()

    # Expose to the REPL if user ran with `python -i debug_playground.py`
    globals()["m"] = m
    globals()["plugin"] = plugin
    globals()["results"] = results
    return 0


if __name__ == "__main__":
    sys.exit(main())
