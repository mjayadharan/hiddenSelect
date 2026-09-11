# R004

Permanent report ID: **R004**
Folder: `2026-09-11-R004-revised-manuscript`
Date: 2026-09-11

Title: Sparse Optimization using Multiple Shooting and Guess Propagation — revised manuscript:
the 2025 draft (`Multishooting_and_sparse_optimization.zip`, `main.tex`) rewritten with the
tightened bounds of R003 (logarithmic norm, residual-based node-removal constant, expectation
over noise, per-window exponents), a numerical results section built from R002, the three
hand-drawn figures replaced by R002's computed figures, and R002's schematic Figure 7 retired.

Supersedes: the 2025 draft manuscript (not a report; kept in the zip). Extends: R002 (numerical
results, figures, tables), R003 (bounds). Inherits protocol from: R002 (data recipe, arms,
metrics), restated in §2 of this report.

## Notes
- No new numerics. All inherited files are byte copies with hashes in `INPUTS.md`; gates G1/G2/G4
  assert byte identity with the source folders. `deps/` empty (nothing imported).
- Corrections to the draft that go beyond constants: comparator of the windowed cost is the
  noise-free windowed cost $J_K^\star$ (not the single-shooting $J^\star$); the per-window data count
  $n_{\max}$ was missing from Proposition 1; the strong-convexity constant is renamed $m$.
