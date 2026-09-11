"""verify_R003.py -- gates for R003 (tighter error bounds). All gates are ordinary
correctness gates (no bug-presence gates). Reads external_data/ and analysis/results/
only; recomputes nothing heavier than a Jacobian. Exit 1 if any gate fails.
Run:  /opt/anaconda3/bin/python3 analysis/verify_R003.py
"""
from __future__ import annotations
import hashlib, json, re, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; RPT = HERE.parent; EXT = RPT / "external_data"; RES = HERE / "results"
GATES = {}
def gate(name, ok, detail=""):
    GATES[name] = {"pass": bool(ok), "detail": str(detail)}; print(f"{name}: {'PASS' if ok else 'FAIL'}  {detail}")

def main():
    # G1 -- inputs match INPUTS.md hashes
    txt = (RPT / "INPUTS.md").read_text(); bad = []
    for m in re.finditer(r"`([^`]+)`\s*\|\s*`([0-9a-f]{64})`", txt):
        f, h = m.group(1), m.group(2)
        if hashlib.sha256((EXT / f).read_bytes()).hexdigest() != h: bad.append(f)
    n = len(re.findall(r"`[0-9a-f]{64}`", txt))
    gate("G1_inputs_match_INPUTS_hashes", n == 7 and not bad, f"{n} files hashed; mismatches: {bad}")

    S = json.loads((RES / "summary.json").read_text()); meta = json.loads((EXT / "concept_meta.json").read_text())
    # G2 -- recomputed constants agree with R002
    rel = lambda a, b: abs(a - b) / abs(b)
    d2 = dict(L=rel(S["L_fine_recomputed"], meta["L_traj"]), mu=rel(S["lognorm_fine_recomputed"], meta["lognorm_traj"]),
              Lbox=rel(S["L_box_recomputed"], meta["L_box"]), mu_box_vs_analytic=abs(S["lognorm_box_recomputed"] - S["lognorm_global_analytic"]))
    gate("G2_constants_match_R002", d2["L"] < 1e-3 and d2["mu"] < 1e-3 and d2["Lbox"] < 1e-9 and d2["mu_box_vs_analytic"] < 1e-6, json.dumps({k: f"{v:.2e}" for k, v in d2.items()}))
    # G3 -- ordering and weighted-norm supremum
    gate("G3_mu_le_L_and_muD_is_one", S["lognorm_traj"] <= S["L_traj"] and abs(S["lognorm_D_global"] - 1.0) < 1e-9,
         f"mu={S['lognorm_traj']:.3f} <= L={S['L_traj']:.3f}; mu_D={S['lognorm_D_global']:.12f}")
    # G4..G7 -- bounds hold on every probe (recomputed here from the raw CSVs, not read from summary)
    L, mu, Lt, mulo = meta["L_traj"], meta["lognorm_traj"], meta["Ltilde_traj"], S["lognorm_lower_fine"]
    l1 = pd.read_csv(EXT / "lemma1_flow_sensitivity.csv"); l2 = pd.read_csv(EXT / "lemma2_param_sensitivity.csv"); tol = 1e-9
    gate("G4_lemma1_original_bounds_hold", ((l1.ratio <= np.exp(L*l1.t)+tol) & (l1.ratio >= np.exp(-L*l1.t)-tol)).all(), f"{len(l1)} probe points")
    gate("G5_lemma1_proposed_bounds_hold", ((l1.ratio <= np.exp(mu*l1.t)+tol) & (l1.ratio >= np.exp(mulo*l1.t)-tol)).all(),
         f"max ratio/upper = {(l1.ratio/np.exp(mu*l1.t)).max():.6f}")
    gate("G6_lemma2_original_bound_holds", (l2.ratio <= Lt/L*(np.exp(L*l2.t)-1)+tol).all(), f"{len(l2)} probe points")
    gate("G7_lemma2_proposed_bound_holds", (l2.ratio <= Lt/mu*(np.exp(mu*l2.t)-1)+tol).all(), f"max ratio/bound = {(l2.ratio/np.maximum(Lt/mu*(np.exp(mu*l2.t)-1),1e-300)).max():.4f}")
    # G8 -- committed tables are byte-identical to a fresh regeneration
    before = {p.name: p.read_bytes() for p in (RPT / "tables").glob("*.tex")}
    subprocess.run([sys.executable, str(HERE / "make_tables.py")], check=True, capture_output=True)
    after = {p.name: p.read_bytes() for p in (RPT / "tables").glob("*.tex")}
    gate("G8_tables_regenerate_byte_identical", before == after, f"{sorted(after)}")

    (RES / "gates_summary.json").write_text(json.dumps(GATES, indent=2))
    npass = sum(g["pass"] for g in GATES.values()); print(f"{npass}/{len(GATES)} gates pass")
    sys.exit(0 if npass == len(GATES) else 1)

if __name__ == "__main__": main()
