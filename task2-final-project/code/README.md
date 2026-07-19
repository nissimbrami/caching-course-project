# Cost-Aware GDSF Eviction Policy for GPTCache

**Author:** Nissim Brami — `nissimbrami@post.bgu.ac.il`
**Course:** Caching in LLMs
**Institution:** Ben-Gurion University of the Negev

This repository contains my final project for the graduate Caching in LLMs course.
It is a cost-aware eviction policy — a **GDSF (Greedy-Dual-Size-Frequency)**
variant — implemented as a drop-in replacement for
[GPTCache](https://github.com/zilliztech/GPTCache)'s default LRU/FIFO
eviction, together with a full benchmark and ablation suite.

## Why this project

LLM API calls are not equal. A short cached prompt might save fractions of a cent;
a long completion with a big model can save several cents per hit. Standard
recency policies like LRU treat every entry the same. My goal was to see how
much of that lost value a well-known cost-aware policy from the CDN literature
(GDSF) recovers when applied to LLM semantic caching, and how sensitive the
result is to the α (frequency) and β (cost) exponents.

## Quick start

```bash
git clone https://github.com/nissimbrami/cost-aware-eviction-gptcache.git
cd cost-aware-eviction-gptcache
docker compose up --build
```

Results (CSVs and plots) appear in `results/`.

Non-Docker alternative:

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
pip install -e .
make test          # 259 tests, ~2s
make benchmark     # full suite, ~10-30 min
make plots         # regenerate the 8 figures
```

## The enhancement in one formula

For every cached entry `e`:

```
priority(e) = L + freq(e)^α · cost(e)^β / size(e)
```

- `L` is the aging clock; it advances to the priority of the last evicted
  item on each eviction, so freshly inserted items always start above it.
- `α` weights access frequency (default 1.0).
- `β` weights the dollar/latency cost of regenerating the entry (default 1.0).
- The size denominator penalizes large entries proportionally.

Eviction picks the entry with the smallest priority — an
`IndexedMinHeap` gives O(log n) push/pop and O(1) key lookup.

## What's in the repo

```
.
├── src/cost_aware_eviction/     # the GDSF implementation
│   ├── eviction_manager.py      #   GDSFEvictionManager (main class)
│   ├── priority_queue.py        #   IndexedMinHeap
│   ├── cost_estimator.py        #   token- and model-price-based cost model
│   ├── config.py                #   GDSFConfig dataclass + validation
│   └── gptcache_plugin.py       #   GPTCache EvictionBase adapter
├── benchmarks/
│   ├── run_all.py               # CLI benchmark driver
│   ├── runner.py                # experiment loop + resource sampling
│   ├── policies.py              # LRU / FIFO / LFU / Random / GDSF wrappers
│   ├── workloads.py             # 6 synthetic workloads
│   └── metrics.py               # BenchmarkResult, MetricsCollector, ResourceSampler
├── tests/                       # 259 tests: unit + integration + property (Hypothesis)
├── scripts/
│   ├── run_all.sh               # one-command reproduction
│   ├── run_ablation.py          # α × β parameter sweep
│   └── generate_plots.py        # figures 1–8
├── results/                     # CSV/JSON output + plots + ablation
├── docs/
│   ├── library-justification.md # why GPTCache
│   └── report.pdf               # the 8–12 page report
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── pyproject.toml
```

## What I measure

Per policy × workload × cache size × run (30 runs per config, seeds pinned):

- **Hit rate** and **cost-weighted hit rate (CWHR)**
- **Dollar savings** (sum of costs of served hits)
- **Latency** — p50, p95, p99 in milliseconds
- **Throughput** — queries / second
- **CPU%** — mean and p95 process CPU utilization (via `psutil`, sampled every 50 ms)
- **RSS** — mean and peak resident-set-size in MB
- **GPU utilization** — N/A: the GPTCache semantic-cache eviction path is CPU-only.
- **Memory overhead** — bytes attributable to the policy's own data structures

CSV and JSON logs land in `results/`. Statistical significance is computed
with paired t-tests + Bonferroni correction and BCa bootstrap 95 % CIs.

## Reproducing the report numbers

```bash
make all           # lint + test + benchmark + plots
python scripts/run_ablation.py --num-runs 10
```

Base seed is `42`; run `i` uses `42 + i`. On a laptop-class CPU the full sweep
takes roughly 15–30 minutes.

## Configuration

Environment variables read by `scripts/run_all.sh`:

| Variable | Default | Meaning |
|---|---|---|
| `CACHE_SIZE` | `1000` | max entries per cache |
| `NUM_RUNS` | `30` | repetitions per (policy, workload, size) |
| `ALPHA` | `1.0` | frequency exponent |
| `BETA` | `1.0` | cost exponent |
| `WORKLOAD` | `high_variance_cost` | workload for quick tests |
| `OUTPUT_DIR` | `results/` | where to write CSV/JSON/plots |

Available workloads: `uniform_cost`, `high_variance_cost`, `zipf_variable_cost`,
`bursty`, `adversarial_lru`, `size_varying`.

## License

MIT. See `LICENSE`.
