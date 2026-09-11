# R002 agent brief (shared context for parallel workers)

Repository: `/Users/manu_jay/git_repos/hiddenSelect` (Julia 1.11.3; project = repo root).
Report folder: `reports/2026-09-11-R002-multishooting-figure-atlas/` — everything you write goes here.

## Hard conventions
- Julia: `/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia --project=<repo root> --startup-file=no <script>`,
  run from the report folder. Before running set `export JULIA_DEPOT_PATH="$TMPDIR/jdepot:$HOME/.julia"`.
- **Add no Julia packages.** Only what `Project.toml` already has (DifferentialEquations, Optim, ForwardDiff, LinearAlgebra, Statistics, Random, StaticArrays, ...).
- **No Makie / no plotting in Julia** (cannot precompile in the sandbox). Julia writes CSV/JSON into
  `analysis/results/`; Python (`/opt/anaconda3/bin/python3`, matplotlib 3.10, pandas, numpy, scipy, PIL) plots.
- Analysis scripts start with `include(joinpath(@__DIR__, "common.jl")); include(joinpath(@__DIR__, "hs_core.jl"))`
  and use `write_csv(path, rows::Vector{NamedTuple})`, `append_csv`, `write_json(path, Dict)` from common.jl.
  `RESULTS` = `analysis/results`. Frozen sources: `deps/` (post-fix) and `deps/prefix/` (pre-fix, commit 44abf4a);
  `depspath(f)`, `prefixpath(f)`, `run_block(path, a, b)` executes a line range verbatim. NEVER read the live tree.
- `Distributed` workers cannot start in the sandbox (no sockets). Parallelise by launching several independent
  processes that take a shard index from ARGS (see `analysis/04_sweeps.jl`) and write `analysis/results/shards/…`.
- Temporary files: `$TMPDIR` only. No `/tmp`.
- Python figure scripts: `import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from style import *`
  then `save(fig, "figXX_name")` → writes `figures/figXX_name.pdf` + `.png`. Use the palette/colours from `figures/style.py`
  (fixed colour per entity; direct labels; one axis per panel — no dual axes; sequential colormaps = one hue).
- Deterministic: fixed seeds, no wall-clock in outputs except explicit timing columns.
- No AI attribution anywhere. Do not commit; the orchestrator commits.

## Key API (analysis/hs_core.jl)
- `FHN_LIB::PolyLib` (dim 2, deg 3, labels `1,w,w²,w³,v,vw,vw²,v²,v²w,v³`), `FHN_P` (true FHN coefficients, 20-vector,
  v̇ block then ẇ block), `make_rhs(lib)` → `rhs!(du,u,p,t)`, `PolyLib(dim, deg; names)`.
- `make_dataset(rhs!, p, y0; T_end, δt_fine, downsample, crop, noise_rel, seed)` → `(data=N×(1+d) with time in col 1, tsall, alldata, clean)`;
  defaults reproduce the repository FHN recipe (N=101, Δt=1, 5 % relative noise, seed 1287436679).
- `ms_loss(x0, p, rhs!, data, δt, S, γ, κ; penalty=:flat|:graded)` — multiple-shooting cost, window size κ (data intervals per window),
  S sub-steps of δt per data interval, smooth-ℓ1 weight γ. Blow-up (|x|>1e3 or NaN) returns 1e3 (flat plateau).
- `make_objective(rhs!, data, δt, S, γ, κ)` → closure over z=[x0; p].
- `run_sweep(rhs!, data, p_true, schedule; seed, mode=:propagate|:best|:reset, optimizer=:nm|:bfgs|:lbfgs, iters, γ, S, penalty, x0_scale, p_init)`
  → `(rows, minimizers::Dict{κ→z}, seed_vector, traces)`; rows have `window_size, J, p_err, score, support_recall, false_positives, wall_s, f_calls, iterations, converged, blowup, seed_p_err`.
- `recovery_score(p, p_true)` fractional recovery score; `simulate(rhs!, x0, p, t_end, δt)` → (ts, X).
- `old_to_new_perm()` permutation from the pre-refactor monomial order (1,v,w,v²,vw,w²,v³,v²w,vw²,w³) to the library order.
