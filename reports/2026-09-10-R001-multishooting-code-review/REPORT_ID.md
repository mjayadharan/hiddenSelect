# R001

Permanent report ID: **R001**
Folder: `2026-09-10-R001-multishooting-code-review`
Date: 2026-09-10

Title: Code review of the `hiddenSelect` multiple-shooting / guess-propagation
model-selection pipeline and its companion theory manuscript, with a headless
re-execution of the numerical pipeline and seven verified findings.

Supersedes: none. Extends: none. Inherits protocol from: none — this is the
first report in this repository, so every definition in §2 is stated here
rather than inherited.

## Notes

- **First report under the toolkit.** `reports/REPORT_STANDARD.md`, `_tools/`,
  `_templates/` and `scripts/` were installed into this repository on
  2026-09-10 from `~/git_repos/research-reporting-toolkit`; `CLAUDE.md` was
  created from `CLAUDE_snippet.md` in the same step. The repository's own
  `.claude/skills/` copy (the optional step in `INSTALL.md`) was **not**
  installed — the sandbox denies writes to that path. The skills resolve
  globally from `~/.claude/skills/`, so nothing is missing locally; only the
  clone-to-another-machine guarantee is.
- **Language adaptation.** The standard's `deps/`-first-on-`sys.path` rule is
  written for Python. This repository is Julia: `deps/` holds the frozen Julia
  sources and `analysis/common.jl` exposes it as `DEPS`, which every analysis
  script resolves first and exclusively. The gate harness stays Python
  (`verify_R001.py`, per the standard's `verify_<id>.py`) and reads only the
  result files the Julia stage writes.
- **Bug-presence gates.** G3, G4, G5 and G7 assert that findings F1–F3 are
  still present in the frozen snapshot. They are expected to **flip to failing**
  once those defects are fixed and `deps/` is re-frozen. See §6 "Gate polarity".
- **Draft history.** An earlier draft of this review shipped on 2026-09-10 as
  flat files `reports/2026-09-10-hiddenSelect-code-review.{tex,md,pdf}` with a
  `reports/repro/` directory, before the toolkit was installed. Those were moved
  into this folder and the ad-hoc `repro/` scripts were replaced by
  `analysis/` + frozen `deps/`; the draft read the **live** sources by line
  range, which the freeze now prevents. No result changed in the move except
  the random-$p$ range in §5, which was re-derived under the gate (G12) and
  corrected from $[0.61, 1000]$ to $[0.0721, 1000]$.

## Scope note — a newer commit exists on `origin/main` (added 2026-09-10, post-review)

This review was conducted against **local** `HEAD` = `8648cde`. After the report
was written, `git fetch` revealed that `origin/main` carries one further commit
the local clone did not have:

> `57e430d` — *"Made all changes to make the loss function, problem setup and
> plotting compatible with generic N-dimensional ode. Not tested yet."*
> (Mon 7 Apr 2025; touches `fhn_model_selection_mod 7.jl`,
> `fhn_model_selection_mod 8.jl`, `plotting_window_size.jl`, and adds
> `plotting_window_size backup.jl`.)

Three of the eight frozen sources are therefore superseded upstream. **No result
in this report is wrong** — every number is a correct measurement of `8648cde`,
and `deps/` pins exactly what was measured — but the *currency* of three findings
and of the §4 version-lineage conclusion is not what the report implies. Checked
directly against `origin/main`:

| Finding | at `8648cde` (reviewed) | at `origin/main` `57e430d` |
|---|---|---|
| F1 `odefun_new` linear truncation | present | **still present** — `mod 6` untouched |
| F2 dropped $w$-residual | mods 5, 6, 7, 8 | **fixed in mod 8** (now `data_loss += norm(state_data[window_inds, :] - sol_mat)^2`); **still present in mods 5, 6, 7** |
| F3(a) `UndefVarError: fhn_u0` | present | **fixed** — line 97 now `fhn_y0` |
| F3(b) `size(data)` / global-`data` shadowing | present | **fixed** — `N = size(data_,1)`, `d = size(data_,2)-1`, reads the argument; data is now a matrix with time in column 1 |
| F4 gradient differentiates `fs_loss` | present | **still present** — mod 8 line 748, mod 7 line 601 |
| F5 stale `animate_contourplots!` signature | present | **still present** — line 497, still `silent=true`, still no `unstable_param` |
| §4 "mod 7 is the latest working copy" | correct for `8648cde` | **superseded** — the mod 8 refactor advanced substantially upstream (though its own commit message says "Not tested yet") |

This note is an **additive documentation retrofit** in the sense of
`reports/REPORT_STANDARD.md` §"Notation self-sufficiency": no result, figure,
number or gate was touched. Per the same standard, an earlier report folder is
never edited in place by a later *study* — re-reviewing `57e430d` is R002's job,
not a rewrite of R001.

**Lesson for the next report's checklist:** fetch before freezing. `deps/` was
frozen from a local clone that was one commit behind its own remote, and nothing
in the standard's current checklist catches that.
