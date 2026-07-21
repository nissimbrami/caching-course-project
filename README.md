# Caching in LLMs — Course Project (Nissim Brami)

BGU graduate course "Caching in LLMs" (Prof. Gil Einziger).
Two deliverables:

## Task 1 — In-class presentation
- Paper: **StreamingLLM** (Xiao et al., ICLR 2024).
- Deck + speaker notes: [`task1-presentation/`](task1-presentation/).

## Task 2 — Research mini-paper + open-source contribution
- Contribution: cost-aware **GDSF** eviction as a drop-in
  `EvictionBase` plugin for [zilliztech/GPTCache](https://github.com/zilliztech/GPTCache).
- Code, tests, benchmarks: [`task2-final-project/code/`](task2-final-project/code/)
  (259 tests, 30-seed benchmark, α×β ablation).
- **Report (canonical, ACM `acmart` sigconf, 7 pages, pdfLaTeX):**
  [`task2-final-project/report-latex/report.pdf`](task2-final-project/report-latex/report.pdf).
  Source: [`report.tex`](task2-final-project/report-latex/report.tex) +
  [`references.bib`](task2-final-project/report-latex/references.bib).
  Overleaf import: [`overleaf-project.zip`](task2-final-project/report-latex/overleaf-project.zip).

## Headline results (paired-t, Bonferroni across 6 workloads, 95% BCa 10k bootstrap)
| Workload            | ΔCWHR (GDSF−LRU) | 95% BCa CI          | paired-t | p_Bonferroni       | $ Δ vs LRU |
|---------------------|-----------------:|:--------------------|---------:|:-------------------|-----------:|
| high_variance_cost  | +0.1190          | [+0.1024, +0.1358]  |   +13.85 | 8.6×10⁻²⁶          | **+25.7%** |
| bursty              | +0.1626          | [+0.1496, +0.1749]  |   +25.31 | 5.9×10⁻⁴⁹          | **+32.3%** |
| adversarial_lru     | +0.1419          | [+0.1003, +0.1894]  |    +6.28 | 3.4×10⁻⁸           | **+18.8%** |
| size_varying        | +0.1632          | [+0.1525, +0.1729]  |   +31.32 | 1.6×10⁻⁵⁸          | **+91.0%** |
| uniform_cost        | +0.00003         | [−0.0004, +0.0005]  |    +0.09 | 1.00 (n.s.)        | +0.005%    |
| zipf_variable_cost  | −0.0003          | [−0.0005, −0.0002]  |    −4.08 | 4.9×10⁻⁴           | −0.037%    |

All numbers resolve to keys in `task2-final-project/code/results/stats_20260721_191113.json`
(produced deterministically by `scripts/compute_statistics.py`, RNG seed `20260721`).

## Reproducibility
```bash
git clone https://github.com/nissimbrami/caching-course-project
cd caching-course-project/task2-final-project/code
pip install -r requirements.txt && pip install -e .
pytest -q                                          # 259 tests
python -m benchmarks.run_all --n-runs 30           # 3,600 experiments
python scripts/compute_statistics.py               # stats JSON
python scripts/run_ablation.py --num-runs 30       # α×β grid
python scripts/generate_plots.py                   # Figures 1–8
```

The LaTeX PDF is auto-built on every push by
[`.github/workflows/build-latex.yml`](.github/workflows/build-latex.yml)
(TeX Live in a container) — downloadable as the `report-pdf` artifact.

## Licence
MIT (code) — see [`task2-final-project/code/LICENSE`](task2-final-project/code/LICENSE).
