# R002 — declared context (Mode A: atlas alongside a fresh computation run)

## Study identity
Multiple shooting with guess propagation for sparse polynomial model selection
(FitzHugh–Nagumo primary; Lotka–Volterra and Lorenz as generality checks), the
`hiddenSelect` repository, theory manuscript `Multishooting_and_sparse_optimization`
(`main.tex`). Purpose of this report: a **figure atlas for a presentation** —
(i) proper concept figures replacing the manuscript's hand sketches,
(ii) fresh simulations supporting the theory's claims, (iii) quantification of the
repository defects found in R001 (and new ones found here) together with their fixes.

## Producing-code premise
All numerics run from the frozen snapshot in `deps/` (post-fix sources, committed with
this report) through `analysis/common.jl` + `analysis/hs_core.jl`; the bug scripts also
execute `deps/prefix/` (pre-fix sources, commit `44abf4a`). Nothing reads the live tree.
The core loss `ms_loss` is gated against the frozen mod 8 `forward_simulation_loss_windows`
(G2, rel. tolerance 1e-10) and the dataset against the frozen mod 8 data block (G3).

## Expected parameter regime (frozen protocol constants)
N = 101 samples, Δt = 1, S = 10 sub-steps (δt = 0.1), 5 % relative noise per component,
data seed 1287436679, γ = 5e-2 (smooth-ℓ1, α = 500), blow-up plateau 1e3, NelderMead
2500 iterations, optimiser seed scale 0.1 (‖p₀ − p*‖ ≈ 1.6), window sizes
FULL = {1,2,3,4,5,6,8,10,12,15,20,25,33,50,75,100}, SHORT = {1,2,5,10,25,50,100}.

## Metric conventions in force
Primary: J_κ (multiple-shooting cost incl. sparsity term) and ‖p − p*‖ (Euclidean);
fractional recovery score (per-coefficient credit, tol_abs 0.05 / tol_rel 25 %);
support recall; false positives; blow-up flag (J ≥ 1e3). Paired differences between
arms sharing an optimiser seed with seed-resampled bootstrap CIs (B = 2000, 95 %).

## Planned grid (realised grid is recomputed by analysis/make_inventory.py)
See `analysis/04_sweeps.jl` (394 jobs), `05_solvers.jl`, `06_bugs.jl`, `07_other_systems.jl`,
plus the deterministic descriptor scripts `01_data.jl`, `02_concept.jl`, `03_landscape.jl`.

## Sanitised user sketch
Figures introducing multiple shooting / guess propagation / node removal (replacing
hand-drawn sketches); trajectories and fits; cost landscapes and their evolution with
window size; guess-propagation vs no-propagation results over several seeds; different
optimisers, solvers, parameters (noise, sparsity, schedule, penalty, sub-steps);
bug findings with before/after figures; an animation of the sweep; a table of figure
captions at the front of the PDF.
