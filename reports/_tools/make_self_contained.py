#!/usr/bin/env python
"""make_self_contained.py — retrofit + enforce the report self-containment
standard (owner mandate 2026-07-17; see reports/REPORT_STANDARD.md).

For every report folder under reports/ this tool (idempotently):
  1. Vendors a frozen snapshot of the shared code modules its scripts import
     into <report>/deps/ (byte-identical copies, verified), with
     deps/MANIFEST.md recording source path, SHA-256, and the git commit at
     vendor time. The closure is the 5 shared modules below; scripts keep
     working against the live tree, but the frozen copy wins (next step) so
     the report regenerates identically even if the shared tree drifts.
  2. Edits each script that imports shared code: after every existing
     `sys.path.insert(0, ...)` / `sys.path.append(...)` line it inserts, at
     the same indentation, a line putting <report>/deps first on sys.path
     (marker-tagged, idempotent, syntax-checked with py_compile).
  3. Writes <report>/INPUTS.md: every external data root/file the report's
     scripts read, with per-file SHA-256 (files) or size + git-tracked count
     (large roots), plus the recorded command that produced the data.
  4. Writes <report>/REPRODUCE.md: pinned environment, ordered regeneration
     commands, and pointers to deps/ and INPUTS.md.

Run from anywhere:  python reports/_tools/make_self_contained.py [--check]
--check verifies (deps byte-identical to their recorded hashes, marker lines
present) without writing.
"""
from __future__ import annotations
import hashlib
import py_compile
import re
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
REPO = TOOLS.parents[1]
RPTS = REPO / "reports"

# ADAPT: the shared modules this repo's report scripts import.  Key = the
# file name as it will appear in <report>/deps/, value = its path relative to
# the repo root.  Leave empty if reports import nothing shared.
# This repository is JULIA. Report scripts import no shared *Python* modules, so
# the automatic Python vendoring below has nothing to do; each report freezes the
# Julia sources it analyses into <report>/deps/ by hand (with a SHA-256
# MANIFEST.md) and exposes them through analysis/common.jl's DEPS constant, which
# every analysis script resolves first and exclusively. See
# reports/2026-09-10-R001-multishooting-code-review/deps/MANIFEST.md.
SHARED = {}
# ADAPT: an import of any SHARED module triggers vendoring for that script.
TRIGGER = re.compile(
    "|".join(rf"\bimport {re.escape(n[:-3])}\b|\bfrom {re.escape(n[:-3])}\b"
             for n in SHARED) or r"(?!x)x")
PATHLINE = re.compile(r"^(\s*)sys\.path\.(?:insert\(0,|append\()")
DEPS_MARK = "# self-contained: frozen deps/ shadows the live shared tree"
DEPS_STMT = ("sys.path.insert(0, str(__import__('pathlib').Path(__file__)"
             ".resolve().parents[1] / 'deps'))  " + DEPS_MARK)

ENV = """- Analysis interpreter: julia 1.11.3 (/Applications/Julia-1.11.app/Contents/Resources/julia/bin/julia), project = repo root Project.toml/Manifest.toml
- Pinned packages: DifferentialEquations 7.15.0, Optim 1.11.0, ForwardDiff 0.10.38, DataFrames 1.7.0
- Gate harness: /opt/anaconda3/bin/python3 (Python 3.11.7, pandas 2.3.3)
- LaTeX: pdflatex (TeX Live 2024), run twice from the report folder
- Frozen code snapshot: `deps/` in this folder (see deps/MANIFEST.md) is a byte copy of the Julia sources analysed; `analysis/common.jl` exposes it as DEPS and every script resolves it first and exclusively, so regeneration uses the frozen code even after the live tree moves on. This is the Julia analogue of the standard's `deps/`-first-on-sys.path rule."""

PY = "/opt/anaconda3/bin/python3"   # gate harness only; the analysis is Julia 1.11.3

# ---------------------------------------------------------------------------
# Per-report configuration: external inputs + reproduction commands.
# data_roots: (path-rel-to-repo, produced-by note) — hashed per file if a
# file, summarized if a directory. reproduce: the ordered command section.
# ---------------------------------------------------------------------------
REPORTS = {
    # ---- TEMPLATE: one entry per report, added when the report is created.
    # data_roots: list of (path-relative-to-repo, "produced by ..." note).
    #   A file is hashed; a directory is summarized (size + tracked count).
    # reproduce : the ordered command block that lands in REPRODUCE.md.
    #
    # "2026-01-01-R001-my-first-report": dict(
    #     data_roots=[("data/raw/run_2026_01", "scripts/run_sweep.py --grid A")],
    #     reproduce=f"""1. `{PY} analysis/run_stage1.py --workers 8`
    # 2. `{PY} figures/make_figures.py`
    # 3. `pdflatex -interaction=nonstopmode -halt-on-error report.tex` (x2)""",
    # ),
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()


def vendor_deps(rpt: Path, check: bool) -> bool:
    deps = rpt / "deps"
    if check:
        return not deps.exists() or all(
            (deps / n).exists() and sha256(deps / n) == sha256(REPO / s)
            for n, s in SHARED.items() if (deps / n).exists())
    deps.mkdir(exist_ok=True)
    lines = ["# deps/ — frozen shared-code snapshot (self-containment standard)",
             "", f"Vendored at git commit `{git_head()}`. These byte-identical",
             "copies shadow the live shared tree on sys.path so this report",
             "regenerates identically even if the shared code later changes.", "",
             "| module | source (repo-relative) | sha256 |", "|---|---|---|"]
    for name, src in SHARED.items():
        s = REPO / src
        (deps / name).write_bytes(s.read_bytes())
        lines.append(f"| {name} | {src} | {sha256(s)} |")
    (deps / "MANIFEST.md").write_text("\n".join(lines) + "\n")
    return True


def edit_scripts(rpt: Path, check: bool) -> list[str]:
    edited = []
    for sub in ("analysis", "figures", "tables"):
        for f in sorted((rpt / sub).glob("*.py")) if (rpt / sub).is_dir() else []:
            text = f.read_text()
            if not TRIGGER.search(text):
                continue
            if DEPS_MARK in text:
                edited.append(f"{f.relative_to(rpt)} (already)")
                continue
            if check:
                edited.append(f"{f.relative_to(rpt)} (MISSING)")
                continue
            out, changed = [], False
            for line in text.splitlines():
                out.append(line)
                m = PATHLINE.match(line)
                if m:
                    out.append(m.group(1) + DEPS_STMT)
                    changed = True
            if changed:
                f.write_text("\n".join(out) + "\n")
                py_compile.compile(str(f), doraise=True)
                edited.append(str(f.relative_to(rpt)))
            else:
                edited.append(f"{f.relative_to(rpt)} (no sys.path lines — "
                              f"uses shared imports via other means; NOT edited)")
    return edited


def write_inputs(rpt: Path, cfg: dict):
    lines = ["# INPUTS.md — external inputs of this report (self-containment standard)",
             "",
             "Everything NOT under this folder that the report's scripts read,",
             "with provenance. In-folder data (results CSVs, external_data/ copies)",
             "is the primary source; the entries below record where it came from",
             "and how to regenerate it.", ""]
    if not cfg["data_roots"]:
        lines.append("None — all inputs are simulated at run time or live in this folder.")
    for root, note in cfg["data_roots"]:
        p = REPO / root
        if p.is_file():
            lines.append(f"- `{root}` — sha256 `{sha256(p)}` — {note}")
        elif p.is_dir():
            files = [q for q in p.rglob("*") if q.is_file()]
            tracked = subprocess.run(
                ["git", "ls-files", root], cwd=REPO, capture_output=True,
                text=True).stdout.splitlines()
            size = sum(q.stat().st_size for q in files)
            lines.append(f"- `{root}/` — {size/1e6:.0f} MB on disk, "
                         f"{len(files)} files ({len(tracked)} git-tracked) — {note}")
            for q in sorted(p.glob("*/manifest.json")) + sorted(p.glob("manifest.json")):
                lines.append(f"    - `{q.relative_to(REPO)}` sha256 `{sha256(q)}` (run provenance)")
        else:
            lines.append(f"- `{root}` — MISSING at standardization time — {note}")
    lines += ["", f"Recorded at git commit `{git_head()}`."]
    (rpt / "INPUTS.md").write_text("\n".join(lines) + "\n")


def write_reproduce(rpt: Path, cfg: dict):
    (rpt / "REPRODUCE.md").write_text(f"""# REPRODUCE.md — {rpt.name}

Self-containment standard (see ../REPORT_STANDARD.md): this folder carries the
report's code, parameters, in-folder results, a frozen snapshot of the shared
code it imports (`deps/`, if any), and this file. Regeneration should need
files predominantly from this folder.

## Environment

{ENV}

## Regeneration steps (in order)

{cfg['reproduce']}

## Inputs and provenance

See `INPUTS.md` (external inputs + hashes) and, where present,
`deps/MANIFEST.md` (frozen shared code) and `analysis/*provenance*`.
""")


def main():
    check = "--check" in sys.argv
    for name, cfg in REPORTS.items():
        rpt = RPTS / name
        if not rpt.is_dir():
            print(f"!! missing report folder: {name}")
            continue
        needs_deps = any(TRIGGER.search(f.read_text())
                         for sub in ("analysis", "figures", "tables")
                         if (rpt / sub).is_dir()
                         for f in (rpt / sub).glob("*.py"))
        deps_ok = vendor_deps(rpt, check) if needs_deps else None
        edits = edit_scripts(rpt, check)
        if not check:
            write_inputs(rpt, cfg)
            write_reproduce(rpt, cfg)
        print(f"== {name}: deps={'n/a' if deps_ok is None else deps_ok}, "
              f"edited={len(edits)}")
        for e in edits:
            print(f"     {e}")


if __name__ == "__main__":
    main()
