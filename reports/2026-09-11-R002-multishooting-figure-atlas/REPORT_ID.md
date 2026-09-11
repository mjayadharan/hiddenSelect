# R002

Permanent report ID: **R002**
Folder: `2026-09-11-R002-multishooting-figure-atlas`
Date: 2026-09-11

Title: Figure atlas for the multiple-shooting / guess-propagation sparse model-selection
pipeline — concept figures, fresh simulations supporting the theory, solver/optimiser/
parameter variations, and quantified repository defects with their fixes.

Supersedes: none. Extends: R001 (code review). Inherits protocol from: R001 — every
inherited definition is restated in §2 of this report.

## Notes
- **Retires R001 gates G3, G4, G5, G7** (bug-presence gates for F1, F2, F3): the
  defects are fixed in the live tree with this report; R002's own gates assert the
  defects are PRESENT in `deps/prefix/` and ABSENT in `deps/`. R001's folder is untouched.
- Julia adaptation of the standard (see R001): `deps/` = frozen Julia sources, exposed
  through `analysis/common.jl`. Two snapshots: `deps/` (post-fix) and `deps/prefix/`.
- `Distributed` cannot start workers in the sandbox; parallel batches are sharded
  processes (`04_sweeps.jl <shard> <n>` + `merge`).
