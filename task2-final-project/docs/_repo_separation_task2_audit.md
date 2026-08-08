# Repo Separation Audit — Task 2 folder (2026-08-08)

**Question**: Does `task2-final-project/` contain exactly what
`INSTRUCTIONS.md` demands for Task 2 (GDSF cost-aware eviction plugin on
GPTCache + report), with no contamination from Task 1?

**Verdict**: **PASS.**

## §9 Stage 3 (code) — GREEN

- One canonical GDSF class: `src/cost_aware_eviction/eviction_manager.py:17 GDSFEvictionManager`
- Adapters: `benchmarks/policies.py:278 GDSFPolicy`, `src/cost_aware_eviction/gptcache_plugin.py:50 GDSFEvictionPlugin`
- Semantic extension: `semantic_gdsf.py:134 SemanticGDSFManager`
- Test suite: **300 passed** in 42.87s (default) / 16.87s (`--hypothesis-seed=0`)
- Property-based tests use hypothesis with fixed seeds

## §9 Stage 4 (results) — GREEN

- `results/benchmark_results_20260721_191113.json`: 3600 rows = 5 policies × 6 workloads × 4 cache sizes × 30 seeds
- `results/stats_20260721_191113.json`: paired-t + BCa 10 000-resample bootstrap, reproduces report Table 5 exactly
- `results/ablation/`: 36 rows = 6 × 6 α × β sweep, non-degenerate
- `results/plots/`: 8 figures in PNG + PDF (only 4 — Figs 2, 3, 5, 8, renumbered as 1-4 — retained in the report; Figs 1, 4, 6, 7 removed from report — data quality issue)

## §9 Stage 5 (report) — GREEN

- `../report-latex/report.pdf`: 9 pages, 740 938 B
- `report.tex` present + bibliography
- 4 figures included via `\includegraphics` (Figs 2, 3, 5, 8 — renumbered as 1-4 in the report; Figs 1, 4, 6, 7 removed from report — data quality issue)
- 0 undefined references, 0 AI-trace hits in compiled portion

## Task 1 contamination check — CLEAN

- Only `StreamingLLM`/Task-1 mention in `task2-final-project/` is SCALM as a
  Related-Work citation at `report.tex:619` — Task-2 relevant, not Task-1 material.
- Code, tests, results, and figures reference only GPTCache / GDSF / LRU / LFU / FIFO — no attention-sink logic.

## Non-blocking observations (optional cleanup)

- Add `.hypothesis/` to `code/.gitignore`
- Consider excluding `docs/H10_*`, `docs/TASK2_*`, `docs/_*_audit.md` from
  the final push (they are internal work-notes, not part of the deliverable)
- `results/archive/` and `results/.smoke/` can be pruned before final commit

None of the above blocks Task 2 delivery.
