# FIGURESET (settled solo — cheap mode)

Primary variation axis: **window size κ** (every result figure sweeps it); secondary axes appear
inside sections (arm, seed, schedule, penalty, noise, γ, optimiser, iterations, S, start scale,
integrator, system). Sections and figures are as in `report.tex` (Parts A–H, figures 1–37).

## Coverage ledger (measured by analysis/make_inventory.py; see planning/coverage_matrix.md)

| unit | realised cells | plotted in body | tables only | justified exclusions |
|---|---|---|---|---|
| sweeps.csv (exp × arm × seed × κ) | 3634 rows / 3634 cells | all experiments have a figure (fig15–28) that shows every (arm, κ) cell as a median and, for main/schedule/noise/lorenz, every seed as a thin line | T1/T2 give κ=100 medians | none |
| other_sweeps.csv | 196 | fig35–36 (every seed drawn) | T5 | none |
| solvers_eval.csv | 168 | fig29 shows p* (56 cells); the two perturbed-p parameter sets are in the CSV only | T3 (p*, κ=5) | perturbed-p cells (112) add no information beyond the p* ones except the κ=25 blow-up noted in the text |
| landscape_2d_* | 8 planes | fig11–12 all | — | none |
| landscape_1d_minima | 84 | fig13 (all in the minima panel; 3 of 12 directions drawn) | — | 9 directions not drawn (summarised in the band) |
| hessian_at_ptrue, post_hessian | 9 + 128 | fig14, fig37 all | — | none |
| lemma1/lemma2 | 24 + 24 probes | fig08–09 all | — | none |
| prop1 | 42 | fig10 all | — | none |
| bug_* | all | fig31–34 all (F2 perturbed-p rows drawn in grey) | T4 | none |

Represented: 100 % of experiment cells (as medians, most also as individual seeds); the only cells not drawn are 112 perturbed-p solver evaluations (in the CSV and text) and 9 of 12 landscape directions (summarised).

Reviewer checks done solo: (a) missing — none (the seed-scale batch was *added* because the main batch exposed the plateau-start failure); (b) unnecessary — none dropped; (c) wrong parameterisation — the first-batch sparsity runs (arm not γ-specific) were superseded and dropped at merge; the animation and filmstrip use seed 2 because seed 1 is the plateau seed, stated in captions.
