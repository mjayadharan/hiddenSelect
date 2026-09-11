# R003 — reproduce

## Environment
```
python   3.11.7 (/opt/anaconda3/bin/python3): numpy 1.26.4, pandas 2.3.3, matplotlib 3.10.9
pdflatex TeX Live 2024, run twice from this folder
pandoc   (not used; report.md is written by hand from report.tex)
```
Set `MPLCONFIGDIR=$TMPDIR/mpl` in the sandbox (the default matplotlib cache dir is read-only).

## Ordered regeneration (all from this folder)
```bash
PY=/opt/anaconda3/bin/python3
$PY analysis/compute_R003.py          # results/*.csv, summary.json            (~2 s)
$PY analysis/make_tables.py           # tables/lemmas.tex, windows.tex, constants.tex
MPLCONFIGDIR=$TMPDIR/mpl $PY figures/make_figures.py   # figures/fig01_bounds.{pdf,png}
$PY analysis/verify_R003.py           # results/gates_summary.json (exit 1 on any failure)
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```
Inputs: see `INPUTS.md` (copies of R002 result files in `external_data/`).
