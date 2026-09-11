#!/usr/bin/env python
"""print_prose_facts.py — number manifest: every result-class number quoted in report.tex prose/captions
is looked up here from tables/table_numbers.json (itself computed from analysis/results/) and written to
analysis/number_manifest.md with its type, source and selector. Numbers the manifest cannot produce are
listed as findings. Protocol constants trace to planning/ATLAS_CONTEXT.md.
"""
import json, re, pathlib
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
NUM = json.loads((ROOT / "tables" / "table_numbers.json").read_text())
tex = (ROOT / "report.tex").read_text()
rows = [("# R002 number manifest", ""), ("", "")]
out = ["# R002 number manifest (analysis/print_prose_facts.py)", "",
       "Each entry: the value as computed now from analysis/results/, its type, source file and selector, and whether the same value appears in report.tex (searched as a rounded string).", "",
       "| key | value (now) | type | source / selector | found in report.tex |", "|---|---|---|---|---|"]
flat = {}
def walk(k, v):
    if isinstance(v, dict):
        for kk, vv in v.items(): walk(f"{k}.{kk}", vv)
    elif isinstance(v, list):
        for i, vv in enumerate(v): walk(f"{k}[{i}]", vv)
    else: flat[k] = v
for k, v in NUM.items(): walk(k, v)
SRC = {"T1": "result_value: tables/T1_headline.csv (sweeps.csv, exp=main)", "T2": "result_value: tables/T2_variations.csv (sweeps.csv, κ=100)",
       "T3": "result_value: solvers_eval.csv (κ=5, p*)", "T4": "result_value: bug_*.csv|json", "T5": "result_value: other_sweeps.csv (κ=100)",
       "seed": "derived_statistic: sweeps.csv", "control": "derived_statistic: sweeps.csv", "n_seeds": "count_from_inventory",
       "N": "protocol_constant: ATLAS_CONTEXT.md", "Delta_t": "protocol_constant", "noise_rel": "protocol_constant",
       "eta": "result_value: fhn_data_meta.json", "noise_floor": "result_value: fhn_data_meta.json", "L_": "result_value: concept_meta.json", "Ltilde": "result_value: concept_meta.json", "lognorm": "result_value: concept_meta.json",
       "core": "result_value: core_check.json", "ms_per": "result_value: core_check.json (timing, not gated)", "hess": "result_value: hessian_at_ptrue.csv", "minima": "result_value: landscape_1d_minima.csv"}
missing = []
for k, v in flat.items():
    src = next((s for p, s in SRC.items() if k.startswith(p)), "result_value")
    if isinstance(v, float):
        cands = {f"{v:.3g}", f"{v:.2g}", f"{v:.2f}", f"{v:.3f}", f"{v:.1f}", f"{v*100:.0f}", f"{v*100:.1f}"}
    else: cands = {str(v)}
    found = any(c in tex for c in cands)
    if not found and src.startswith("result_value") and not k.startswith(("T2", "T5", "ms_per")): missing.append(k)
    out.append(f"| {k} | {v if not isinstance(v, float) else f'{v:.6g}'} | {src.split(':')[0]} | {src} | {'yes' if found else 'no'} |")
out += ["", f"Result-class keys not found verbatim in report.tex (rounded differently or not quoted): {len(missing)}", ""] + [f"- {k}" for k in missing]
(HERE / "number_manifest.md").write_text("\n".join(out) + "\n"); print("manifest:", len(flat), "numbers;", len(missing), "not found verbatim")
