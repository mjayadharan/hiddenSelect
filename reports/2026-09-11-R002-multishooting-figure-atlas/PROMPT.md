# PROMPT.md — build spec for R002 (frozen after planning, before building)

Mission: a figure atlas for a presentation on multiple shooting with guess propagation (hiddenSelect),
with concept figures replacing the manuscript's hand sketches, fresh parallel simulations supporting the
theory, variations (schedule, penalty, noise, sparsity, optimiser, iterations, sub-steps, start scale,
integrator, system), quantified + fixed repository defects, an animation, and a front table of figures.

Deliverables: report.tex/.pdf/.md, figures/fig01–37.{pdf,png}, figures/anim_sweep.gif, tables/T1–T5,
analysis/verify_R002.py → gates_summary.json, number_manifest.md, planning/{ATLAS_CONTEXT,CANDIDATE_INPUTS,
FIGURESET,coverage_matrix}.md, DATA_BINDINGS.json, deps/ (post-fix) + deps/prefix/ (pre-fix) with MANIFEST.md,
REPRODUCE.md, INPUTS.md, REPORT_ID.md.

Data sources: none external; everything computed here from deps/ (Mode A). Binding conventions: notation
self-sufficiency (§2 before results), fractional recovery score + paired bootstrap CIs, pdflatex twice,
no AI attribution, no new Julia packages, no Makie, $TMPDIR only, deterministic seeds.

Planning-actions log: Phase 0 folder + deps freeze (deps/prefix from 44abf4a after git fetch; deps/ after the
fixes); Phase 1 inventory by analysis/make_inventory.py (realised); Phase 1.5 no external inputs; Phase 2-cheap
figure set (planning/FIGURESET.md); Phase 3 bindings (planning/DATA_BINDINGS.json); user approval for new
compute: the request itself ("run multiple simulations in parallel").

Deviations log:
- Distributed workers unavailable in the sandbox → sharded processes with resume keys (04_sweeps.jl).
- Two shards stalled on L-BFGS jobs; the remaining jobs were re-sharded (batches b, c); merge dedupes.
- Sparsity arms renamed to carry γ (first-batch runs superseded and dropped at merge).
- Seed-scale experiment added after the main batch exposed the plateau-start failure.
- fig18/19 and the animation use seed 2 (seed 1 is the plateau seed); fig37 cross-evaluation likewise.
