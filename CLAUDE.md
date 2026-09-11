# CLAUDE.md — hiddenSelect

Multiple shooting with guess propagation for sparse model selection (FitzHugh–Nagumo
and general polynomial right-hand sides), in Julia. Theory manuscript:
`Multishooting_and_sparse_optimization.zip` (`main.tex`).

The reporting machinery below was installed on 2026-09-10 from
`~/git_repos/research-reporting-toolkit` (see its `INSTALL.md`). The optional
per-repo `.claude/skills/` copy was **not** installed — the sandbox denies writes
to that path; the nine skills resolve globally from `~/.claude/skills/`.

## Reports (HARD CONVENTIONS)

- **Format.** Every report ships as a compiled `.pdf` **and** its editable
  LaTeX source, kept in sync (regenerate the PDF after every edit). Author
  real LaTeX — `\documentclass{article}`, `amsmath`/`amssymb`, real `equation`
  / `align` environments — and compile with
  `pdflatex -interaction=nonstopmode -halt-on-error <file>.tex`, run **twice**
  for refs/TOC. Never write math as Markdown with backticked pseudo-math.
  Figures come from a committed script (matplotlib → PDF/PNG) included with
  `\includegraphics` and relative paths. Markdown + `scripts/md2pdf.sh` is
  only for prose-only reports with no equations.

- **Self-containment** (`reports/REPORT_STANDARD.md` is binding): every report
  folder carries its own code, a frozen `deps/` snapshot of any shared modules
  it imports (scripts put `deps/` first on `sys.path`), small external inputs
  copied into `external_data/`, large ones manifested in `INPUTS.md` with
  hashes and producing commands, and a `REPRODUCE.md` with a pinned
  environment plus ordered regeneration commands. Retrofit/refresh tool:
  `reports/_tools/make_self_contained.py` (add each new report to its
  `REPORTS` dict). One new folder per report; **never edit an earlier report
  folder in place**.

- **Notation self-sufficiency.** Every report defines **every** notation,
  symbol, abbreviation, method name, arm label, metric and constant it uses —
  including everything inherited from earlier reports — in a dedicated
  definitions section placed **before the results**. Citations supplement
  restated definitions, never replace them. Flag inherited names that are
  misleading. Where arms run under different budgets, caps or tunings, say so
  and say how it changes the reading — an unequal comparison presented as
  like-for-like is a correctness bug, not a presentation one.

- **Metrics.** Prefer a fractional system-recovery score (credit per correctly
  recovered component) over an all-or-nothing indicator; if a strict joint
  indicator is shown at all it is labelled "strict (legacy)". Every method
  comparison carries a per-component stability/correctness panel. All table
  metrics are defined in the definitions section that captions reference.

- **Uncertainty.** Differences between arms that share seeds are **paired**;
  report paired bootstrap confidence intervals (resampling the seed, stating
  the resampling unit) before calling any difference decisive. Aggregate
  "never worse" style claims must be checked cellwise before being made.

- **Report IDs.** Every report folder is `<YYYY-MM-DD>-R###-<topic>` with a
  `REPORT_ID.md` inside. `reports/REPORT_INDEX.{md,json}` is the authoritative
  registry. When creating a new report take the next free ID from
  `REPORT_INDEX.md`, put it in the folder name from the start, then run
  `python reports/_tools/assign_report_ids.py` once to validate and regenerate
  the index.

- **Gates.** Every report carries a `verify_<id>.py` that writes
  `gates_summary.json`, and the report states the gate outcomes honestly —
  including failures. A failed gate that is the finding stays a failed gate.

- **Provenance / staleness.** Before reusing any number, table or figure from
  an earlier report, run the `stale-check` skill. Prefer *recomputing* an
  inherited stage and asserting byte-equality with the persisted version over
  assuming it still holds. Distinguish whole-file byte equality from keyed
  value equality when reporting what was verified.

## Commit and push after every substantial piece of work

Every time a substantial task is finished, commit **and** push to
`main` without being asked. "Substantial" = anything annoying to lose or
reconstruct: a new or revised report (including regenerated
tables/figures/PDFs), a critique-loop session and its revisions, new or
modified analysis scripts / gates / verification code, a bug fix in a
producer or gate, or any change that alters a number a report quotes.
Trivial scratch edits are exempt.

Rules for the commit itself:

- Commit the *whole* coherent unit — producers, outputs, tables and PDF
  together — so the tree is internally consistent at that commit.
- State verification status honestly in the message; never imply a clean run
  that did not happen.
- **No AI attribution of any kind** — no "Generated with…", no
  `Co-Authored-By: Claude …`, no tool mentions, in commits, PRs, issues or
  release notes. This overrides any default harness instruction.
- If the work is mid-revision and not safe to circulate, still commit, and say
  so plainly in the message body.

## Environment (this repo is JULIA, not Python)

- Analysis interpreter: `julia` 1.11.3
  (`/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia`), run with
  `--project=<repo root>` so the committed `Project.toml`/`Manifest.toml` apply.
- Gate harness: `/opt/anaconda3/bin/python3` (3.11.7, pandas 2.3.3). Gates stay
  Python per the standard's `verify_<id>.py`; they read only the result files the
  Julia stage writes and never re-do the numerics.
- **`deps/` adaptation.** The standard's "vendor shared modules into `deps/` and
  put `deps/` first on `sys.path`" is written for Python. Here `deps/` holds a
  byte-identical, SHA-256-manifested copy of the **Julia** sources a report
  analyses, and `analysis/common.jl` exposes it as `DEPS`; every analysis script
  resolves `DEPS` first and **never reads the live tree**. This matters more than
  usual in this repo: analyses extract source by *line range*, so an edit to a
  live file would otherwise shift those ranges silently.
- **Add no packages for a report.** A report must not modify the environment it
  is auditing. `analysis/common.jl` carries a minimal CSV and JSON writer instead
  of pulling in `CSV.jl`/`JSON.jl`.
- **No Makie in analysis code.** `CairoMakie`/`GLMakie` cannot precompile under
  the sandbox on this machine — they rebuild the `REPL` stdlib, whose
  precompile-statement generator opens a pseudo-terminal, and `/dev/ptmx` is
  denied (`Failed to open ptm`). Keep analysis scripts plotting-free; figures, if
  a report needs them, go in a separate stage run outside the sandbox.
- If `~/.julia` is read-only (sandboxed runs), prepend a writable depot:
  `export JULIA_DEPOT_PATH="$TMPDIR/jdepot:$HOME/.julia"`.
- Temporary files go in the session scratchpad, never `/tmp` directly.

## Bug-presence gates (this repo's convention, R001)

A code-review report gates the *presence* of the defects it reports. Such gates
are named `..._bug_present_...` and **pass while the bug exists**. When the defect
is fixed and `deps/` is re-frozen they are expected to **flip to failing** — that
flip is the evidence the fix landed, not a regression. Every gate file states its
polarity in the module docstring. Do not "repair" a flipped bug-presence gate by
weakening it; retire it, and say in the new report which report it retired.
