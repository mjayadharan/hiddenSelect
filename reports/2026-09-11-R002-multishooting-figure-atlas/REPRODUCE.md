# R002 — reproduce

## Environment

```
julia    1.11.3   (/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia)
project  hiddenSelect/Project.toml + Manifest.toml as committed (no package added)
         DifferentialEquations 7.15.0, Optim 1.11.0, ForwardDiff 0.10.38
python   3.11.7   (/opt/anaconda3/bin/python3): pandas 2.3.3, numpy 1.26.4, matplotlib 3.10.9, Pillow 10.2.0, scipy 1.17.1
pdflatex TeX Live 2024, run twice from this folder
node     (optional) only for the palette validator of the dataviz skill; not needed to regenerate anything
```

Machine notes (same as R001): the analysis loads **no plotting stack** in Julia (CairoMakie cannot
precompile in the sandbox); figures are matplotlib. If `~/.julia` is read-only, prepend a writable depot:
`export JULIA_DEPOT_PATH="$TMPDIR/jdepot:$HOME/.julia"`. `Distributed` workers cannot start in the sandbox
(no sockets), so the two parallel batches are **sharded processes** (below). Temporary files go to `$TMPDIR`.

## Ordered regeneration (all commands from this folder)

```bash
J="/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia --project=../.. --startup-file=no"
PY=/opt/anaconda3/bin/python3
export JULIA_DEPOT_PATH="$TMPDIR/jdepot:$HOME/.julia"

$J analysis/00_check_core.jl        # core-vs-frozen identities + timing   (~15 s)
$J analysis/01_data.jl              # datasets (FHN, LV)                    (~10 s)
$J analysis/02_concept.jl           # concept segments, Lemma 1/2, Prop. 1  (~1 min)
$J analysis/03_landscape.jl         # 2-D / 1-D landscapes, Hessians        (~3 min)
# parallel batch 1 (12 shards) and batch 2 (8 shards; seed-scale + sparsity), then merge
rm -rf analysis/results/shards
for i in $(seq 0 11); do $J analysis/04_sweeps.jl $i 12 > analysis/results/04_shard$i.log 2>&1 & done; wait
R002_TAG=b R002_ONLY_EXPS=seedscale,sparsity bash -c 'for i in $(seq 0 7); do '"$J"' analysis/04_sweeps.jl $i 8 > analysis/results/04b_shard$i.log 2>&1 & done; wait'
$J analysis/04_sweeps.jl merge      # -> sweeps.csv, sweep_minimizers.csv, sweep_traces.csv, sweep_fits.csv
$J analysis/05_solvers.jl           # integrator/solver comparison          (~1.5 min)
$J analysis/06_bugs.jl              # defect quantification (pre- vs post-fix) (~6 min)
for i in 0 1; do $J analysis/07_other_systems.jl $i 2 > analysis/results/07_shard$i.log 2>&1 & done; wait
$J analysis/07_other_systems.jl merge
$J analysis/08_post.jl              # Hessians at the GP minimisers, paths  (~2 min)
$J analysis/09_traces.jl            # NM traces for seed 2                  (~1 min)

$PY analysis/make_inventory.py      # planning/coverage_matrix.{md,json}, results_hashes.json
$PY tables/make_tables.py           # tables/T*.tex|md|csv, table_numbers.json
$PY figures/concept_figures.py      # fig01–fig07 (concept schematics)
$PY figures/make_figures.py         # fig08–fig37 (result figures)
$PY figures/make_animation.py       # anim_sweep.gif + frames
$J analysis/10_animation_data.jl    # landscape at every κ, concept segments, LV fits (~2 min)
$J analysis/11_lv_landscape.jl      # LV landscapes (3 planes) at every κ (~8 min; pass a plane name to do one)
$PY figures/make_animations.py      # nine MP4s (uses the FFMPEG_jll artifact binary, see script header)
$PY slides/make_deck.py             # slides/multishooting_story.pptx ; python slides/qa_geometry.py for the layout check
$PY analysis/verify_R002.py         # gates -> analysis/results/gates_summary.json
$PY analysis/print_prose_facts.py   # analysis/number_manifest.md
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
```

Wall clock on the reference machine (16 cores): batch 1 ≈ 20 min (BFGS jobs dominate), batch 2 ≈ 6 min,
everything else ≈ 15 min serial. Batch 1 alone with a single process would take ≈ 3.5 h.

## Determinism

All RNG streams are seeded in code: data noise `Xoshiro(1287436679)` (FHN; identical to `Random.seed!`
in the frozen mod 8 block — gate G3), LV `2024`, Lorenz `77`; optimiser starts `MersenneTwister(seed)`
per job; landscape directions `MersenneTwister(5)`; basin directions `MersenneTwister(99)`. Wall-time
columns (`wall_s`, `time_ms`) are the only non-deterministic values and are never used in a gate.
