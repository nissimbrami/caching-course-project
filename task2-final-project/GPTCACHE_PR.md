# GPTCache upstream PR — draft body

**Target repo:** `zilliztech/GPTCache` (Apache-2.0, ~12k stars)
**Fork:** https://github.com/nissimbrami/GPTCache
**Branch:** `feat/gdsf-cost-aware-eviction` (pushed, 3 files, +389 lines)
**One-click open URL:**
https://github.com/zilliztech/GPTCache/compare/main...nissimbrami:GPTCache:feat/gdsf-cost-aware-eviction?expand=1

*(The fine-grained PAT can push branches but cannot open PRs across repositories
via the API. Opening the PR is a single click on the URL above; use the
title and body below.)*

---

## Title
`feat(eviction): add cost-aware GDSF policy`

## Body

## Summary

Adds Greedy-Dual-Size-Frequency (GDSF) as a new eviction policy under
`EvictionBase("memory", policy="GDSF", ...)`, alongside the existing
LRU/LFU/FIFO/RR options.

## Motivation

The current in-memory policies are recency- or frequency-only. For LLM
response caches, per-entry regeneration cost varies by 10x–100x between
short GPT-3.5 answers and long GPT-4 completions, and response sizes vary
similarly. Recency-only eviction burns real dollars.

GDSF (Cherkasova, HP Labs 1998; Cao & Irani, USITS 1997) folds frequency,
cost, and size into a single priority score:

```
Priority(i) = L + freq(i)^alpha * cost(i)^beta / size(i)
```

Cao & Irani proved Greedy-Dual is competitive-optimal (ratio *k*) for
weighted caching among deterministic online algorithms.

## Empirical results

Across 30 seeds × 6 workloads × 4 cache sizes = 3,600 runs, with paired-t
tests, Bonferroni correction, and 95% BCa bootstrap CIs (10k resamples,
seed pinned to `20260721`):

| Workload | Δ CWHR (GDSF−LRU) | 95% BCa CI | Bonferroni p | Dollar delta |
|---|---|---|---|---|
| high-variance cost | +0.1190 | [+0.102, +0.136] | 8.6e-26 | **+25.7%** |
| bursty | +0.1626 | [+0.150, +0.175] | 5.9e-49 | **+32.3%** |
| size-varying | +0.1632 | [+0.153, +0.173] | 1.6e-58 | **+91.0%** |
| adversarial anti-LRU | +0.1419 | [+0.100, +0.189] | 3.4e-8 | **+18.8%** |
| uniform cost (control) | +0.00003 | [-0.0004, +0.0005] | 1.00 | +0.005% |
| Zipf, cache fits set | -0.0003 | [-0.0005, -0.0002] | 4.9e-4 | -0.037% |

No statistically meaningful degradation on the two workloads where
cost-aware ranking cannot help. Full benchmark harness, ablation study,
statistics script, and a 259-test suite are open-source at:

https://github.com/nissimbrami/caching-course-project

## API

* **Backwards-compatible**: `put(keys)` still works (defaults
  `cost = size = 1`, reducing GDSF to LFU-with-clock-aging).
* **New extension**: `put_with_metadata([(key, cost, size), ...])` unlocks
  the full cost-aware behaviour.
* O(log n) via an indexed min-heap.
* Thread-safe via a single `RLock`.
* Monotone-clock invariant preserved (`L` only ever grows).

## Tests

8 new unit tests in `tests/unit_tests/manager/test_gdsf_eviction.py`:
- factory wires `policy="GDSF"` → `GDSFEviction`
- `put(objs)` API compatibility
- `get()` increments frequency and updates priority
- cost-aware ranking (higher cost wins ties)
- size penalisation (larger loses ties)
- monotone clock invariant
- frequency-boost tie-breaking
- input validation (positive size, missing-key `get`)

## Context

This PR is the open-source deliverable for a graduate "Caching in LLMs"
project at Ben-Gurion University of the Negev. Full write-up (7-page ACM
`acmart` sigconf paper):
https://github.com/nissimbrami/caching-course-project/blob/main/task2-final-project/report-latex/report.pdf

Marked as **draft** for maintainer review of API design choices (in
particular: `put_with_metadata` vs extending the base `put` signature).
Happy to iterate.
