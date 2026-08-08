# LaTeX Report Source

Professional ACM `acmart` (sigconf) typesetting for the Task 2 report.

## Files
- `report.tex` — main manuscript (all sections, tables, figures, equations)
- `references.bib` — 19 references in BibTeX format
- `figures/` — 4 vector PDF figures produced by `code/scripts/generate_plots.py`
  (source files retain original IDs `fig2`, `fig3`, `fig5`, `fig8`; renumbered
  as Fig 1–4 in the report body)
- `report.pdf` — pre-built PDF (8 pages, ~654 KB)

## Build

### Option 1 — GitHub Actions (already configured)
Every push to `main` triggers `.github/workflows/build-latex.yml`, which
compiles the PDF in a full TeX Live container and uploads it as the
`report-pdf` artifact. Download from the workflow run page.

### Option 2 — Local (requires TeX Live / MikTeX)
```bash
cd task2-final-project/report-latex
latexmk -pdf report.tex
```
`latexmk` will run the `pdflatex → bibtex → pdflatex → pdflatex` sequence
automatically. A manual sequence works equivalently.

### Option 3 — Overleaf
Zip `report.tex`, `references.bib`, and `figures/`, upload to Overleaf,
select pdfLaTeX as compiler with `report.tex` as the main document:
```bash
cd task2-final-project/report-latex
zip -r report-source.zip report.tex references.bib figures/
```

## Numeric-claim provenance
Every statistic in Section 5 of `report.tex` resolves to a key in
`task2-final-project/code/results/stats_20260721_191113.json`, which is
produced deterministically by
`task2-final-project/code/scripts/compute_statistics.py` (bootstrap RNG
seed `20260721`).
