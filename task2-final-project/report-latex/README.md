# LaTeX Report Source

Professional ACM `acmart` (sigconf) typesetting for the Task 2 report.

## Files
- `report.tex` — main manuscript (all sections, tables, figures, equations)
- `references.bib` — 14 references in BibTeX format
- `figures/` — 8 vector PDF figures produced by `scripts/generate_plots.py`
- `overleaf-project.zip` — self-contained Overleaf import (rebuild with `make overleaf-zip` or the command below)

## Build

### Option 1 — Overleaf (recommended, zero setup)
1. Upload `overleaf-project.zip` to Overleaf (New Project → Upload Project).
2. Compiler: **pdfLaTeX**. Main document: `report.tex`.
3. Recompile → produces `report.pdf`.

### Option 2 — GitHub Actions (already configured)
Every push to `main` triggers `.github/workflows/build-latex.yml`, which
compiles the PDF in a full TeX Live container and uploads it as the
`report-pdf` artifact. Download from the workflow run page.

### Option 3 — Local (requires TeX Live / MikTeX)
```bash
cd task2-final-project/report-latex
latexmk -pdf report.tex
```

## Rebuild the Overleaf zip
```bash
cd task2-final-project/report-latex
zip -r overleaf-project.zip report.tex references.bib figures/
```

## Numeric-claim provenance
Every statistic in Section 5 of `report.tex` resolves to a key in
`task2-final-project/code/results/stats_20260721_191113.json`, which is
produced deterministically by
`task2-final-project/code/scripts/compute_statistics.py` (bootstrap RNG
seed `20260721`).
