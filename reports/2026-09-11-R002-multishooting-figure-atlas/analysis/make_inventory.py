#!/usr/bin/env python
"""make_inventory.py — realised coverage matrix + SHA-256 ledger of every result file.

Scans analysis/results/*.csv|json and writes planning/coverage_matrix.{md,json} and
analysis/results/results_hashes.json. Deterministic; no numbers are invented — the
axes and cell counts are read from the files.
"""
import hashlib, json, sys
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent; RES = HERE / "results"; PLAN = HERE.parent / "planning"
AXES = {  # file -> axis columns defining the atomic coverage unit
    "sweeps.csv": ["exp", "arm", "seed", "window_size"],
    "other_sweeps.csv": ["system", "arm", "optimizer", "seed", "window_size"],
    "solvers_eval.csv": ["param", "window_size", "method", "setting"],
    "solvers_sweep.csv": ["method", "seed", "window_size"],
    "landscape_2d_v_v3.csv": ["window_size"], "landscape_2d_w_wv.csv": ["window_size"],
    "landscape_1d_minima.csv": ["direction", "window_size"], "hessian_at_ptrue.csv": ["window_size"],
    "prop1_node_removal.csv": ["param", "experiment", "n_removed"],
    "lemma1_flow_sensitivity.csv": ["start_index", "direction"], "lemma2_param_sensitivity.csv": ["start_index", "direction"],
    "bug_F2_w_residual.csv": ["window_size", "param"], "bug_F4_gradient.csv": ["window_size"],
    "post_hessian_at_minimizers.csv": ["seed", "window_size"],
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

files = sorted([p for p in RES.iterdir() if p.suffix in (".csv", ".json")])
ledger = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size} for p in files}
(RES / "results_hashes.json").write_text(json.dumps(ledger, indent=1, sort_keys=True))

out = ["# Realised coverage matrix (computed by analysis/make_inventory.py)", "",
       "Atomic unit per file = one row of the listed axes; `cells` = distinct axis tuples, `rows` = file rows.", ""]
cov = {}
for name, axes in AXES.items():
    p = RES / name
    if not p.exists():
        out.append(f"## {name}\n\n**missing** (planned, not realised)\n"); cov[name] = None; continue
    df = pd.read_csv(p)
    ax_present = [a for a in axes if a in df.columns]
    cells = df[ax_present].drop_duplicates()
    out.append(f"## {name}  (rows {len(df)}, cells {len(cells)}, sha256 {ledger[name]['sha256'][:12]}…)\n")
    info = {"rows": int(len(df)), "cells": int(len(cells)), "axes": {}}
    for a in ax_present:
        vals = sorted(df[a].dropna().unique().tolist(), key=lambda v: (str(type(v)), v))
        info["axes"][a] = [v if not hasattr(v, "item") else v.item() for v in vals]
        vs = ", ".join(str(v) for v in vals[:40]) + (" …" if len(vals) > 40 else "")
        out.append(f"- `{a}` ({len(vals)} values): {vs}")
    if "exp" in df.columns and "arm" in df.columns:
        out.append("\n| exp | arm | seeds | window sizes | rows |\n|---|---|---|---|---|")
        for (e, a), g in df.groupby(["exp", "arm"]):
            out.append(f"| {e} | {a} | {g.seed.nunique()} | {g.window_size.nunique()} | {len(g)} |")
        info["exp_arm"] = {f"{e}/{a}": {"seeds": int(g.seed.nunique()), "window_sizes": int(g.window_size.nunique()), "rows": int(len(g))}
                           for (e, a), g in df.groupby(["exp", "arm"])}
    if "system" in df.columns and "arm" in df.columns:
        out.append("\n| system | optimizer | arm | seeds | window sizes | rows |\n|---|---|---|---|---|---|")
        for (s, o, a), g in df.groupby(["system", "optimizer", "arm"]):
            out.append(f"| {s} | {o} | {a} | {g.seed.nunique()} | {g.window_size.nunique()} | {len(g)} |")
    out.append(""); cov[name] = info
(PLAN / "coverage_matrix.md").write_text("\n".join(out))
(PLAN / "coverage_matrix.json").write_text(json.dumps(cov, indent=1, default=str))
print("inventory:", len(files), "result files;", sum(1 for v in cov.values() if v), "of", len(AXES), "axis files realised")
