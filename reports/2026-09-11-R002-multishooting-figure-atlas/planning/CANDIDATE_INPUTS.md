# Candidate inputs (Mode A — all inputs are produced inside this folder)

No external result files are consumed. Every result file under `analysis/results/` is
written by an in-folder script from the frozen `deps/` snapshot; realised files and
their SHA-256 are listed by `analysis/make_inventory.py` in `planning/coverage_matrix.md`
and `analysis/results/results_hashes.json`. The three hand-drawn sketches in the theory
manuscript (`shooting_partition.png`, `multiple_shooting.png`,
`comparison_shooting_nodes.png`) were consulted as design references only and are not
inputs to any figure. Earlier report R001's numbers are NOT reused; every quantity that
overlaps R001 (the headline sweep, the F1/F2 quantification) is recomputed here.

| planned output | producing script | role | status |
|---|---|---|---|
| fhn_*.csv, lv_*.csv, fhn_data_meta.json | 01_data.jl | primary | realised |
| concept_*.csv, lemma1/2_*.csv, prop1_*.csv | 02_concept.jl | primary | realised |
| landscape_*.csv, hessian_*.csv | 03_landscape.jl | primary | realised |
| sweeps.csv, sweep_minimizers.csv, sweep_traces.csv, sweep_fits.csv | 04_sweeps.jl (12 shards + merge) | primary | pending at planning time |
| solvers_*.csv | 05_solvers.jl | primary | pending |
| bug_*.csv/json | 06_bugs.jl | primary (defect quantification) | pending |
| other_*.csv, lorenz_*.csv | 07_other_systems.jl (2 shards + merge) | primary | pending |
| post_*.csv | 08_post.jl | derived (from sweep minimisers) | pending |
