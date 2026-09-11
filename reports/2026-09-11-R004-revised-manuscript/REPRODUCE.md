# R004 — reproduce

## Environment
```
python   3.11.7 (/opt/anaconda3/bin/python3): numpy 1.26.4, pandas 2.3.3
pandoc   3.x (/opt/homebrew/bin/pandoc) for report.md
pdflatex TeX Live 2024, run twice from this folder (packages: amsmath, amsthm, mathtools, algorithm, algpseudocode, needspace, booktabs, hyperref)
```
R004 runs no numerics. The inherited files (figures, tables, result files) are byte copies of R002/R003 outputs;
to regenerate *them*, follow `REPRODUCE.md` of those reports.

## Ordered regeneration (all from this folder)
```bash
PY=/opt/anaconda3/bin/python3
$PY analysis/make_inputs_md.py     # INPUTS.md (hashes + sources of every inherited file)
$PY analysis/make_tables.py        # tables/ (verbatim copies, T2_compact, numbers.tex/json, md twins)
$PY analysis/verify_R004.py        # gates -> analysis/results/gates_summary.json, tables/gates_status.tex (G9/G10 fail until md/pdf exist)
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
$PY analysis/make_report_md.py     # report.md
$PY analysis/verify_R004.py        # final gate run (all 10 expected to pass)
pdflatex -interaction=nonstopmode -halt-on-error report.tex   # re-inline the final gate status
```
