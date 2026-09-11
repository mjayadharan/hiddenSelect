# Report self-containment standard

Owner mandate (2026-07-17): every report folder under `reports/` must carry
everything needed to regenerate or modify that report **predominantly from
files inside the folder itself**, so the setup of a report is never lost even
if the shared code or sibling reports change later. This applies to all
existing reports (retrofitted by `reports/_tools/make_self_contained.py`) and
to every report going forward.

## Required contents of a report folder

1. **Report-specific code in-folder.** All analysis/aggregation, figure, and
   table scripts live under `<report>/analysis|figures|tables/`. No report
   logic lives outside the folder.
2. **Frozen shared-code snapshot: `deps/`.** If any script imports shared
   repo modules (`bench_common`, `dae_finder`, `common`, `sweep_common`,
   `helper_functions`), byte-identical copies are vendored into
   `<report>/deps/` with `deps/MANIFEST.md` (source path, SHA-256, git commit
   at vendor time). Every such script puts `deps/` FIRST on `sys.path`
   (marker line `# self-contained: frozen deps/ shadows the live shared
   tree`), so regeneration uses the frozen code even if the live tree drifts.
3. **Input data in-folder when small; manifested when large.** Result CSVs
   produced by the report's own scripts live in-folder. Small external inputs
   (roughly ≤ 25 MB, e.g. a sibling report's CSVs) are COPIED into the folder
   (`external_data/`) and the scripts read the local copy. Large inputs
   (multi-GB benchmark results roots) stay at their committed in-repo
   location and are recorded in `INPUTS.md` with size, git-tracked file
   count, per-file hashes where feasible, and the exact command that produced
   them.
4. **`REPRODUCE.md`.** Pinned environment (interpreter path, Python and key
   package versions), the ordered regeneration commands from data →
   analysis → figures → tables → `pdflatex` (x2), and the report's key
   parameters where they are not obvious from the scripts.
5. **`INPUTS.md`.** Every external input with provenance (see 3).
6. **Provenance for re-analyses.** A report that re-aggregates another
   report's outputs records the SHA-256 of its inputs at aggregation time
   (e.g. `provenance.json` written by the aggregation script).

## Notation self-sufficiency (owner mandate 2026-07-28)

Every report must be readable without any other document. **Every notation,
symbol, abbreviation, short form, method name, arm label, metric and protocol
constant it uses must be defined inside it** — including everything inherited
from earlier reports.

Each report therefore carries a **dedicated definitions section, before the
results**, restating what it reuses (first applied in
`2026-07-28-R047-equivalence-graded-comparison/`, §2 "Inherited definitions"):

1. **Methods / arms** — one row per arm actually used, defined in full, not by
   citation. Composite arm names (`rule | anchor | rank-rule`) get each slot
   explained and each option enumerated.
2. **Metrics and flags** — including ones merely reused (`in_class`, `minimal`,
   fractional system score, strict legacy, audited stability …), with the
   primary metric identified.
3. **Symbols and constants** — symbol → meaning → value → where frozen.
   Explicitly separate look-alikes (a *method's* cap `K_MAX` vs a *metric's*
   budget `κ`; a stopping rule's `τ` vs any metric tolerance).
4. **Provenance per item** — which report defined it. The citation is
   *additional to* the restated definition, never a substitute for it.

Two things must be surfaced, not silently reproduced:

- **Misleading inherited names.** If an arm's label does not describe what it
  computes, say so (R047's `S3_splithalf` selects on held-out MSE over a
  trajectory-wise 6/2 train/validation split, not on agreement between two
  halves).
- **Non-uniform configurations.** If arms run under different caps, budgets,
  tunings or libraries, state it and say how it changes the reading. In R047
  the three design-based classicals are capped at 2 terms, the QR arms at 2 or
  4, and STLSQ/SR3 are uncapped — which is *why* the latter score 0.000 at
  `equiv_k(4)`. Presenting unequal configurations as like-for-like is a
  correctness defect, not a presentation one.

Applies to reports already shipped as well: bringing an existing report up to
this standard is an **additive documentation retrofit** (a new definitions
section, no result touched, gates unchanged) and is a deliberate narrow
exception to "earlier report folders are never edited in place" below — that
rule exists to stop later *studies* overwriting earlier *findings*. Record the
retrofit in the report's `REPORT_ID.md`. Retrofitted so far: R047 (as authored),
R038 (2026-07-28).

Where a definition claim is checkable against the shipped data (term caps,
support sizes, selection rules), gate it. Where it is a property of a
*producer* the report does not run, cite the source file and line and say it is
a citation rather than writing a gate that cannot fail.

## Report IDs (owner mandate 2026-07-23)

Every report folder carries a permanent unique ID embedded in its name
directly after the date: `<YYYY-MM-DD>-R###-<topic>`. IDs are assigned
chronologically, never reused or renumbered, and exist so results can be
compiled and cited across reports by ID. Each folder also contains a
`REPORT_ID.md` (id, date, slug, former folder name), and
`reports/REPORT_INDEX.{md,json}` is the authoritative registry.

When creating a NEW report: take the next free ID (highest in
`REPORT_INDEX.md` + 1) and put it in the folder name from the start, then
run `reports/_tools/assign_report_ids.py` once — it validates, writes
`REPORT_ID.md`, and regenerates the index. If a folder was created
without an ID, the same tool assigns one, renames the folder (git mv when
tracked), and rewrites repo-internal references to the old name.
`--check` verifies without writing.

## Tooling

`reports/_tools/make_self_contained.py` retrofits/refreshes items 2, 4, 5
idempotently for all configured reports (`--check` verifies without writing).
When creating a NEW report: add its entry to the tool's `REPORTS` dict
(inputs + reproduce commands) and run the tool once; copy small external
inputs into `external_data/` and point the scripts at the local copy.
`reports/_tools/assign_report_ids.py` maintains the report-ID registry
(section above).

## Related standing conventions (defined elsewhere, applied per report)

- Fractional system-exact metric as the default recovery metric, strict joint
  as labeled legacy, per-equation stability/correctness panel in every method
  comparison (first applied in `2026-07-17-R011-dominant-balance-eqscore/`).
- Explicit metric-definitions section referenced by every table caption.
- Stale-check discipline for any report that reuses previous results
  (recompute from source, record hashes, audit prose claims including
  captions/figure titles/quantifiers).
- One new folder per report; earlier report folders are never edited in
  place by later studies.
