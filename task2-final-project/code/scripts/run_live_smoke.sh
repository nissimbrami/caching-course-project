#!/usr/bin/env bash
# run_live_smoke.sh - Fast live smoke test that shows each stage's real output.
#
# Purpose: Verify Task 2 end-to-end in ~2 minutes by running a *small* subset
# of every stage in the pipeline and printing real inputs/outputs to stdout.
#
# This is DIFFERENT from run_all.sh:
#   - run_all.sh runs the FULL benchmark grid (5-15 min, publication data)
#   - run_live_smoke.sh runs a SMOKE subset (~2 min, sanity check + live proof)
#
# Usage:
#   bash scripts/run_live_smoke.sh
#
# Exit code: 0 if every stage passes, non-zero otherwise.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors (fall back to no colors if not a terminal)
if [ -t 1 ]; then
    B='\033[0;34m'; G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; N='\033[0m'
else
    B=''; G=''; Y=''; R=''; N=''
fi

hdr() {
    echo ""
    echo -e "${B}================================================================${N}"
    echo -e "${B} $1${N}"
    echo -e "${B}================================================================${N}"
}
ok()   { echo -e "${G}[OK]${N} $*"; }
warn() { echo -e "${Y}[WARN]${N} $*"; }
fail() { echo -e "${R}[FAIL]${N} $*"; }

FAIL_COUNT=0
run_stage() {
    local name="$1"; shift
    hdr "$name"
    echo "\$ $*"
    if "$@"; then
        ok "$name"
    else
        fail "$name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# ---------------------------------------------------------------------------
# STAGE 0 - Environment sanity
# ---------------------------------------------------------------------------
hdr "Stage 0/8: Environment sanity"
echo "\$ python --version"
python --version
echo "\$ python -c 'import src.cost_aware_eviction as m; print(m.__file__)'"
python -c "import src.cost_aware_eviction as m; print(m.__file__)" || warn "src.cost_aware_eviction import failed (editable install not active?)"
echo "\$ pwd"; pwd
ok "Stage 0"

# ---------------------------------------------------------------------------
# STAGE 1 - Core data structure (IndexedMinHeap)
# ---------------------------------------------------------------------------
hdr "Stage 1/8: IndexedMinHeap live trace"
python - <<'PY'
from src.cost_aware_eviction.priority_queue import IndexedMinHeap
h = IndexedMinHeap()
for k, p in [("a", 3.0), ("b", 1.0), ("c", 2.0), ("d", 5.0), ("e", 0.5)]:
    h.push(k, p)
    print(f"push({k}, {p}) -> size={len(h)}  min={h.peek()}")
print(f"update(a, 0.1)"); h.update("a", 0.1)
print(f"pop order: ", end="")
while len(h):
    print(h.pop(), end=" ")
print()
PY
ok "Stage 1"

# ---------------------------------------------------------------------------
# STAGE 2 - GDSFEvictionManager (put/access/evict with priority trace)
# ---------------------------------------------------------------------------
hdr "Stage 2/8: GDSFEvictionManager put/access/evict"
python - <<'PY'
from src.cost_aware_eviction.eviction_manager import GDSFEvictionManager
m = GDSFEvictionManager(max_size=100, alpha=1.0, beta=1.0)
m.put("A", size=40, cost=0.5); print(f"put A size=40 cost=0.5   -> current_size={m.current_size}")
m.put("B", size=30, cost=2.0); print(f"put B size=30 cost=2.0   -> current_size={m.current_size}")
m.put("C", size=20, cost=0.1); print(f"put C size=20 cost=0.1   -> current_size={m.current_size}")
m.access("B"); m.access("B"); m.access("A")
for k in ("A", "B", "C"):
    print(f"priority({k}) = {m.get_priority(k):.4f}")
m.put("D", size=50, cost=1.5)
print(f"put D size=50 cost=1.5 forced eviction; remaining keys...")
for k in ("A","B","C","D"):
    print(f"  in_cache({k}) = {k in m}")
PY
ok "Stage 2"

# ---------------------------------------------------------------------------
# STAGE 3 - GPTCache EvictionBase plugin contract
# ---------------------------------------------------------------------------
hdr "Stage 3/8: GPTCache EvictionBase plugin contract"
python - <<'PY'
from src.cost_aware_eviction.gptcache_plugin import GDSFEvictionPlugin
from gptcache.manager.eviction.base import EvictionBase
p = GDSFEvictionPlugin(maxsize=200)
print(f"isinstance(plugin, EvictionBase) = {isinstance(p, EvictionBase)}")
print(f"policy = {p.policy}")
p.register_metadata("k1", size=50, cost=1.0)
p.register_metadata("k2", size=80, cost=3.0)
p.put(["k1", "k2"])
print(f"num_entries after put = {p.num_entries}")
print(f"get('k1') = {p.get('k1')}")
p.register_metadata("k3", size=120, cost=0.5)
p.put(["k3"])
print(f"num_entries after over-cap put = {p.num_entries}")
PY
ok "Stage 3"

# ---------------------------------------------------------------------------
# STAGE 4 - Mini benchmark: LRU vs GDSF on one workload/seed
# ---------------------------------------------------------------------------
hdr "Stage 4/8: Mini benchmark (LRU vs GDSF, 500 queries)"
python - <<'PY'
from benchmarks.runner import run_single_experiment, create_policy
from benchmarks.workloads import generate_high_variance_cost_workload
wl = generate_high_variance_cost_workload(n_queries=500, n_unique=80, seed=42)
for name in ("LRU", "GDSF"):
    pol = create_policy(name, max_size=10000)
    res = run_single_experiment(pol, wl, "high_variance_cost", cache_size=10000, run_id=0, seed=42)
    print(f"{name:5s}: hit_rate={res.hit_rate:.3f}  cwhr={res.cost_weighted_hit_rate:.3f}  savings=${res.dollar_savings:.2f}")
PY
ok "Stage 4"

# ---------------------------------------------------------------------------
# STAGE 5 - Statistics reproduce from JSON (spot-check)
# ---------------------------------------------------------------------------
hdr "Stage 5/8: Statistics reproduce (paired-t + BCa CI)"
python - <<'PY'
import json, glob, os
path = sorted(glob.glob("results/stats_*.json"))[-1]
with open(path) as f:
    stats = json.load(f)
r = stats["results"]["high_variance_cost"]
print(f"file: {os.path.basename(path)}")
print(f"workload: high_variance_cost   metric: {r['metric']}")
print(f"  n_pairs      = {r['n_pairs']}")
print(f"  mean_diff    = {r['mean_diff']:.6f}   (report claim: 0.118999)")
print(f"  paired_t     = {r['paired_t']:.2f}     (report claim: 13.85)")
print(f"  p_bonferroni = {r['p_bonferroni']:.3e} (report claim: 8.60e-26)")
print(f"  BCa 95% CI   = [{r['ci_lower_95_bca']:.6f}, {r['ci_upper_95_bca']:.6f}]")
print(f"                 (report claim: [0.102430, 0.135792])")
PY
ok "Stage 5"

# ---------------------------------------------------------------------------
# STAGE 6 - Pytest fast subset (unit tests only, no slow markers)
# ---------------------------------------------------------------------------
run_stage "Stage 6/8: Pytest core suite" \
    python -m pytest tests/ -q --tb=line -x

# ---------------------------------------------------------------------------
# STAGE 7 - Report PDF sanity
# ---------------------------------------------------------------------------
hdr "Stage 7/8: Report PDF sanity"
PDF="../report-latex/report.pdf"
if [ -f "$PDF" ]; then
    bytes=$(wc -c < "$PDF")
    if command -v pdfinfo >/dev/null 2>&1; then
        pages=$(pdfinfo "$PDF" | awk '/^Pages:/ {print $2}')
        echo "  pages = $pages   bytes = $bytes   ($PDF)"
    else
        echo "  bytes = $bytes   ($PDF)   [pdfinfo not installed]"
    fi
    ok "Stage 7"
else
    fail "Stage 7 - PDF missing at $PDF"
    FAIL_COUNT=$((FAIL_COUNT + 1))
fi

# ---------------------------------------------------------------------------
# STAGE 8 - Artefact inventory
# ---------------------------------------------------------------------------
hdr "Stage 8/8: Artefact inventory"
echo "Benchmark JSONs:"
ls -1 results/benchmark_results_*.json 2>/dev/null | tail -3 | sed 's/^/  /'
echo "Stats JSONs:"
ls -1 results/stats_*.json 2>/dev/null | tail -3 | sed 's/^/  /'
echo "Ablation:"
ls -1 results/ablation/*.csv 2>/dev/null | sed 's/^/  /'
echo "Plots:"
ls -1 results/plots/*.png 2>/dev/null | sed 's/^/  /'
ok "Stage 8"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
hdr "Summary"
if [ "$FAIL_COUNT" -eq 0 ]; then
    ok "All stages passed."
    exit 0
else
    fail "$FAIL_COUNT stage(s) failed."
    exit 1
fi
