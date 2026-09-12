#!/usr/bin/env python
"""verify_R002.py — the gate harness for R002. Reads ONLY this folder's own result files
(analysis/results/, tables/, deps/MANIFEST.md, figures/) and writes analysis/results/gates_summary.json.

Polarity: gates named `..._bug_present_...` assert that a defect is PRESENT in the frozen
PRE-fix snapshot (deps/prefix/) and, where the name says so, ABSENT in the post-fix snapshot
(deps/). They are expected to keep passing for as long as deps/prefix/ is the 44abf4a snapshot.
Every other gate is an ordinary correctness/completeness gate whose failure means something broke
or did not run to completion — and a failure is reported as such, never repaired here.
Run: /opt/anaconda3/bin/python3 analysis/verify_R002.py   (exit 1 if any gate fails)
"""
import hashlib, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent; RES = HERE / "results"; DEPS = ROOT / "deps"
GATES = {}
def gate(name, ok, detail=""):
    GATES[name] = {"pass": bool(ok), "detail": str(detail)}; print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
def rd(n): return pd.read_csv(RES / n)
def js(n): return json.loads((RES / n).read_text())

# G1 frozen sources match the manifest
man = (DEPS / "MANIFEST.md").read_text(); ok = True; n = 0
for m in re.finditer(r"\| `([^`]+)` \| `([0-9a-f]{64})` \|", man):
    p = DEPS / m.group(1); n += 1
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != m.group(2): ok = False
gate("G1_deps_frozen_matches_manifest", ok and n == 12, f"{n} files checked")

# G2–G4 core identities
c = js("core_check.json")
gate("G2_core_loss_equals_frozen_mod8_loss", c["loss_max_reldiff"] < 1e-10, f"max rel diff {c['loss_max_reldiff']:.2e} over 7 κ × 6 p")
gate("G3_dataset_equals_frozen_mod8_block", c["dataset_max_absdiff"] < 1e-12, f"max abs diff {c['dataset_max_absdiff']:.2e}")
gate("G4_rhs_identities", c["rhs_vs_odefun_poly_max"] < 1e-12 and c["rhs_vs_old_order_odefun_max"] < 1e-12,
     f"vs odefun_poly! {c['rhs_vs_odefun_poly_max']:.1e}; vs old-order odefun (permuted) {c['rhs_vs_old_order_odefun_max']:.1e}")

# G5–G9 bug presence (pre-fix) and absence (post-fix)
b = js("bug_summary.json")
gate("G5_F1_bug_present_prefix_linear_truncation_absent_fixed",
     b["F1_J11_prefix_at_uref"] == 1.0 and abs(b["F1_J11_fixed_at_uref"] + 1.25) < 1e-12 and b["F1_max_abs_fixed_minus_exact"] < 1e-12,
     f"J11 prefix {b['F1_J11_prefix_at_uref']} vs fixed {b['F1_J11_fixed_at_uref']} (exact -1.25); max|fixed-exact| {b['F1_max_abs_fixed_minus_exact']}")
f2 = rd("bug_F2_w_residual.csv"); f2t = f2[f2.param == "ptrue"]
gate("G6_F2_bug_present_prefix_drops_w_residual_absent_fixed",
     f2t.missing_fraction.between(0.05, 0.25).all() and (abs(f2t.J_fixed_stiff - f2t.J_explicit) < 1e-8).all(),
     "missing at p*: " + ", ".join(f"{v*100:.1f}%" for v in f2t.sort_values('window_size').missing_fraction) + f"; max|fixed-explicit| {(f2t.J_fixed_stiff - f2t.J_explicit).abs().max():.1e}")
f4 = rd("bug_F4_gradient.csv").set_index("window_size")
gate("G7_F4_bug_present_prefix_gradient_is_single_shooting", f4.cosine_similarity.loc[1] < 0.5 and f4.cosine_similarity.loc[100] > 0.999,
     f"cos at κ=1: {f4.cosine_similarity.loc[1]:.3f}, κ=5: {f4.cosine_similarity.loc[5]:.3f}, κ=100: {f4.cosine_similarity.loc[100]:.4f}")
om = js("bug_ordering_meta.json")
gate("G8_ordering_bug_present_prefix_mod8_driver", om["spurious_p_err"] > 1.0 and om["J_old_order_k1"] > 10*om["J_correct_order_k1"],
     f"spurious ‖p−p*‖ {om['spurious_p_err']:.3f}; J(p*) old-order/correct at κ=1: {om['J_old_order_k1']:.3f}/{om['J_correct_order_k1']:.4f}")
gate("G9_shadowing_bug_present_prefix_absent_fixed", b["shadowing_prefix_identical"] and b["shadowing_fixed_differs"], "pre-fix J identical for 3 datasets; post-fix differs")

# G10–G13 main sweep
sw = rd("sweeps.csv"); m = sw[sw.exp == "main"]
gate("G10_main_sweep_complete", len(m) == 3*8*16 and m.groupby("arm").seed.nunique().eq(8).all(), f"{len(m)} rows (expected 384)")
piv = m.pivot_table(index=["seed", "window_size"], columns="arm", values="p_err").reset_index()
tie1 = (piv[piv.window_size == 1].propagate == piv[piv.window_size == 1].reset).all()
better = piv[piv.window_size > 1].assign(b=lambda x: x.propagate < x.reset).groupby("window_size").b.sum()
plateau_seeds = m[(m.arm == "propagate") & (m.window_size == 1) & m.blowup].seed.nunique()
gate("G11_GP_better_than_control_cellwise", tie1 and (better >= 8 - plateau_seeds).all(),
     f"tied at κ=1: {tie1}; GP strictly better in ≥{8 - plateau_seeds}/8 seeds at every κ≥2 (min {better.min()}); seeds starting on the plateau: {plateau_seeds}")
t1 = pd.read_csv(ROOT / "tables" / "T1_headline.csv").set_index("window_size")
gate("G12_paired_CI_excludes_zero_at_k100", t1.ci_hi.loc[100] < 0, f"GP−control at κ=100: {t1.paired_diff.loc[100]:.3f} [{t1.ci_lo.loc[100]:.3f}, {t1.ci_hi.loc[100]:.3f}]")
ctrl = m[(m.arm == "reset") & m.blowup]
gate("G13_control_plateau_is_the_seed", (abs(ctrl.p_err - ctrl.seed_p_err) < 1e-12).all() and len(ctrl) > 0, f"{len(ctrl)} control rows on the plateau all return the seed")

# G14 solver identity
se = rd("solvers_eval.csv"); s10 = se[(se.param == se.param.unique()[0]) & se.setting.astype(str).str.contains("S=10")]
gate("G14_inhouse_S10_matches_adaptive_reference", (s10.abs_err_vs_reference < 1e-8).all(), f"max |J−J_ref| {s10.abs_err_vs_reference.max():.1e} over κ∈{sorted(s10.window_size.unique())}")

# G15 other systems, paired per seed
o = rd("other_sweeps.csv"); o100 = o[o.window_size == 100]
ok = True; det = []
for (s, op), g in o100.groupby(["system", "optimizer"]):
    pv = g.pivot(index="seed", columns="arm", values="p_err"); w = int((pv.propagate < pv.reset).sum()); ok &= w == len(pv); det.append(f"{s}/{op}: {w}/{len(pv)}")
gate("G15_other_systems_GP_better_every_seed", ok, "; ".join(det))

# G16 planned vs realised jobs
meta = js("sweeps_meta.json")
gate("G16_all_planned_sweep_jobs_realised", meta["n_jobs_done"] >= meta["n_jobs_planned"], f"{meta['n_jobs_done']} done of {meta['n_jobs_planned']} planned")

# G17 every figure referenced by report.tex exists as PDF; results ledger present
tex = (ROOT / "report.tex").read_text() if (ROOT / "report.tex").exists() else ""
figs = re.findall(r"\\includegraphics\[[^\]]*\]\{figures/([^}]+)\}", tex)
missing = [f for f in figs if not (ROOT / "figures" / f).exists()]
gate("G17_figures_and_ledger_present", len(figs) > 0 and not missing and (RES / "results_hashes.json").exists(), f"{len(figs)} figures referenced; missing: {missing}")

# G18 tables reproduce from results (T1 recomputed here vs the shipped CSV)
rows = []
for k, g in piv.groupby("window_size"):
    rows.append((k, g.propagate.median(), g.reset.median()))
chk = pd.DataFrame(rows, columns=["window_size", "gp", "ctrl"]).set_index("window_size")
gate("G18_T1_reproduces", np.allclose(chk.gp, t1.perr_GP) and np.allclose(chk.ctrl, t1.perr_control), "median ‖p−p*‖ per κ recomputed = tables/T1_headline.csv")

# G19-G20 extended landscape animations (Part I). Ordinary completeness/correctness gates:
# G19 asserts the announced animation set exists; G20 asserts that the basin/plateau numbers in
# tables/T7 (and printed in every frame) are recomputable from the shipped grids, so the videos and
# the table cannot quote different geometry.
FULL_LADDER = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100]
ANIM2D = [f"anim_fhn_landscape_{k}_hires.mp4" for k in ("v_v3", "wv_ww", "v2_w2", "w_wv", "v3_w3")] + \
         ["anim_lv_landscape_x2_xy_hires.mp4"]
ANIM3D = [f"anim3d_fhn_landscape_{k}.mp4" for k in ("wv_ww", "v2_w2", "v_v3")] + \
         ["anim3d_lv_landscape_x2_xy.mp4", "anim3d_fhn_landscape_wv_ww_control.mp4"]
if (RES / "landscape_plane_screen_summary.csv").exists():
    sc = rd("landscape_plane_screen_summary.csv")
    missing_mp4 = [f for f in ANIM2D + ANIM3D
                   if not (ROOT / "figures" / f).exists() or (ROOT / "figures" / f).stat().st_size < 10_000]
    gate("G19_extended_landscape_animations_present",
         len(sc) == 16 and int(sc.selected.sum()) == 5 and not missing_mp4,
         f"{len(sc)} candidate planes screened, {int(sc.selected.sum())} animated; "
         f"{len(ANIM2D)} 2-D + {len(ANIM3D)} 3-D MP4s; missing/empty: {missing_mp4}")

    if (RES / "landscape_hires_geometry.csv").exists():
        hg = rd("landscape_hires_geometry.csv"); ok = True; det = []
        meta = js("landscape_planes_meta.json"); meta.update(
            {"lv_" + k: v for k, v in js("landscape_lv_hires_meta.json").items()})
        for plane, g in hg.groupby("plane"):
            m = meta[plane]; n = int(m["n"])
            df = pd.read_csv(RES / m["file"])
            if len(df) != n * n * len(FULL_LADDER):
                ok = False; det.append(f"{plane}: {len(df)} rows"); continue
            J = df.J.values.reshape(len(FULL_LADDER), n, n)
            gi = g.set_index("window_size")
            for ki, k in enumerate(FULL_LADDER):
                Z = J[ki]; fin = Z < 1e3
                if abs((1 - fin.mean()) - gi.f_blow.loc[k]) > 1e-9 or \
                   abs((Z < 2 * Z[fin].min()).mean() - gi.basin.loc[k]) > 1e-9:
                    ok = False; det.append(f"{plane}@k={k}")
            det.append(f"{plane}: {n}x{n}x{len(FULL_LADDER)} ok" if ok else "")
        gate("G20_landscape_geometry_recomputes_from_grids", ok and len(hg) == 6 * len(FULL_LADDER),
             f"{len(hg)} (plane, kappa) cells re-derived from the shipped grids; " + "; ".join(d for d in det if d))

(RES / "gates_summary.json").write_text(json.dumps(GATES, indent=1))
npass = sum(g["pass"] for g in GATES.values()); print(f"\n{npass}/{len(GATES)} gates pass")
failed = [k for k, g in GATES.items() if not g["pass"]]
(ROOT / "tables" / "gates_status.tex").write_text(f"{npass} of {len(GATES)} gates pass" + (" (failing: " + ", ".join(k.replace("_", "\\_") for k in failed) + ")" if failed else "") + "\n")
sys.exit(0 if npass == len(GATES) else 1)
