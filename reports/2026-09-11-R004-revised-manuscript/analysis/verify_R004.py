"""verify_R004.py -- gates for R004 (revised manuscript). All are ordinary correctness/inheritance
gates; none is a bug-presence gate. Reads this folder, the R002/R003 folders (for byte identity)
and the copied result files; recomputes nothing heavier than an exponential.
Writes analysis/results/gates_summary.json and tables/gates_status.tex. Exit 1 if any gate fails.
Run: /opt/anaconda3/bin/python3 analysis/verify_R004.py
"""
from __future__ import annotations
import hashlib, json, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd
from sources import RPT, inherited_files, source_of
RES = RPT / "analysis" / "results"; RES.mkdir(exist_ok=True); EXT = RPT / "external_data"; TAB = RPT / "tables"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
GATES = {}
def gate(name, ok, detail=""):
    GATES[name] = {"pass": bool(ok), "detail": str(detail)}; print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")

def main():
    tex = (RPT / "report.tex").read_text()
    # G1 -- external_data/ files: hash in INPUTS.md and byte-identical to source
    inputs = (RPT / "INPUTS.md").read_text(); listed = dict(re.findall(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", inputs))
    bad = [rel for rel in inherited_files() if rel.startswith("external_data/") and (listed.get(rel) != sha(RPT / rel) or sha(source_of(rel)[0]) != sha(RPT / rel))]
    n_ext = sum(r.startswith("external_data/") for r in inherited_files())
    gate("G1_external_data_hashes_and_byte_identity", n_ext > 0 and not bad, f"{n_ext} files; mismatches: {bad}")
    # G2 -- figures byte-identical to source
    figs = [r for r in inherited_files() if r.startswith("figures/")]
    badf = [r for r in figs if listed.get(r) != sha(RPT / r) or sha(source_of(r)[0]) != sha(RPT / r)]
    gate("G2_figures_byte_identical_to_source", len(figs) == 20 and not badf, f"{len(figs)} figures; mismatches: {badf}")
    # G3 -- every referenced figure exists; no retired figure referenced
    refs = re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
    missing = [r for r in refs if not (RPT / r).exists()]
    retired = [n for n in ("shooting_partition.png", "multiple_shooting.png", "comparison_shooting_nodes.png", "fig07_error_growth") if re.search(r"includegraphics\[[^\]]*\]\{[^}]*" + re.escape(n), tex)]
    gate("G3_referenced_figures_exist_no_retired", refs and not missing and not retired and len(set(refs)) == 20, f"{len(refs)} references, missing={missing}, retired={retired}")
    # G4 -- verbatim tables byte-identical to their sources
    vb = {"T1_headline.tex": EXT / "tables" / "T1_headline.tex", "T5_other_systems.tex": EXT / "tables" / "T5_other_systems.tex", "lemmas.tex": EXT / "tables" / "lemmas.tex", "windows.tex": EXT / "tables" / "windows.tex", "constants.tex": EXT / "tables" / "constants.tex"}
    badt = [k for k, v in vb.items() if sha(TAB / k) != sha(v)]
    gate("G4_verbatim_tables_byte_identical", not badt, f"{list(vb)}; mismatches: {badt}")
    # G5 -- generated tables regenerate byte-identically
    gen = ["numbers.tex", "numbers.json", "T2_compact.tex", "T2_compact.md", "lemmas.md", "windows.md"]
    before = {g: (TAB / g).read_bytes() for g in gen}
    subprocess.run([sys.executable, str(RPT / "analysis" / "make_tables.py")], check=True, capture_output=True)
    after = {g: (TAB / g).read_bytes() for g in gen}
    gate("G5_generated_tables_regenerate_byte_identical", before == after, f"{gen}")
    # G6 -- prose numbers agree with R002 table_numbers.json / R003 summary.json (to printed precision)
    N = json.loads((TAB / "numbers.json").read_text()); TN = json.loads((EXT / "tables" / "table_numbers.json").read_text()); S3 = json.loads((EXT / "tables" / "R003_summary.json").read_text())
    checks = {"perrGPfinal": TN["T1_final_perr_GP"], "perrCtrlFinal": TN["T1_final_perr_control"], "pairedDiff": TN["T1_final_paired_diff"], "ciLo": TN["T1_final_ci"][0], "ciHi": TN["T1_final_ci"][1],
              "bfgsGP": TN["T2_optimizer"]["bfgs_propagate"][0], "schedDense": TN["T2_schedule"]["dense"][0], "peakAmp": S3["lemma1_peak_amplification"], "mu": S3["lognorm_traj"], "L": S3["L_traj"], "weightS": S3["weight_s"]}
    bad6 = [k for k, v in checks.items() if abs(N[k] - v) > 5e-3 * max(1, abs(v))]
    gate("G6_prose_numbers_match_sources", not bad6, f"{len(checks)} numbers checked; mismatches: {bad6}")
    # G7 -- revised lemma constants never exceed the draft's (FHN constants)
    L, mu, Lt = S3["L_traj"], S3["lognorm_traj"], S3["Ltilde_traj"]; ok7 = True
    for s in (1, 2, 5, 10):
        ok7 &= np.exp(mu*s) <= np.exp(L*s) and Lt/mu*(np.exp(mu*s)-1) <= Lt/L*(np.exp(L*s)-1)
    gate("G7_revised_constants_le_draft", ok7, f"mu={mu:.3f} <= L={L:.3f}; both Lemma factors compared at s in (1,2,5,10)")
    # G8 -- revised Lemma 1 / Lemma 2 bounds hold at every probe (recomputed from the copied CSVs)
    l1 = pd.read_csv(EXT / "lemma1_flow_sensitivity.csv"); l2 = pd.read_csv(EXT / "lemma2_param_sensitivity.csv"); tol = 1e-9; mulo = S3["lognorm_lower_fine"]
    ok8 = bool(((l1.ratio <= np.exp(mu*l1.t)+tol) & (l1.ratio >= np.exp(mulo*l1.t)-tol)).all() and (l2.ratio <= Lt/mu*(np.exp(mu*l2.t)-1)+tol).all())
    gate("G8_revised_bounds_hold_on_probes", ok8, f"{len(l1)} + {len(l2)} probe points")
    # G9 -- report.md carries every section heading, table and figure of report.tex
    md = (RPT / "report.md").read_text() if (RPT / "report.md").exists() else ""
    secs = re.findall(r"\\(?:sub)?section\{([^}]+)\}", tex); nfig_tex = tex.count("\\begin{figure}"); ntab_tex = tex.count("\\begin{table}")
    missing_secs = [s_ for s_ in secs if s_.split("(")[0].strip().rstrip(":").split(" ")[0] not in md]
    nfig_md = len(re.findall(r"^\*\*Figure \d+\*\*", md, re.M)); ntab_md = len(re.findall(r"^\*\*Table \d+\.\*\*", md, re.M))
    gate("G9_markdown_complete", md and not missing_secs and nfig_md == nfig_tex and ntab_md == ntab_tex, f"sections missing={missing_secs}; figures md/tex={nfig_md}/{nfig_tex}; tables md/tex={ntab_md}/{ntab_tex}")
    # G10 -- pdf exists, newer than tex, page count
    pdf = RPT / "report.pdf"; pages = None
    if pdf.exists():
        m = re.search(r"Output written on report.pdf \((\d+) pages", (RPT / "report.log").read_text(errors="ignore")); pages = int(m.group(1)) if m else None
    gate("G10_pdf_built", pdf.exists() and pdf.stat().st_mtime >= (RPT / "report.tex").stat().st_mtime and pages, f"pages={pages}")
    (RES / "gates_summary.json").write_text(json.dumps(GATES, indent=2))
    npass = sum(g["pass"] for g in GATES.values()); (TAB / "gates_status.tex").write_text(f"{npass} of {len(GATES)} gates pass")
    print(f"{npass}/{len(GATES)} gates pass"); sys.exit(0 if npass == len(GATES) else 1)
if __name__ == "__main__": main()
