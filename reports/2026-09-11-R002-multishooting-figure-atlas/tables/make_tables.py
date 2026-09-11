#!/usr/bin/env python
"""make_tables.py — every table in the report is generated here from analysis/results/.
Writes tables/*.tex (LaTeX fragments, booktabs) and tables/*.md (Markdown twins used by report.md),
plus tables/table_numbers.json (the numbers quoted in the prose, for the number manifest).
"""
import json, sys, pathlib
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; RES = HERE.parent / "analysis" / "results"
def rd(n): return pd.read_csv(RES / n)
def js(n): return json.loads((RES / n).read_text())
NUM = {}

def write(name, df, caption_note="", fmt=None, index=False, colspec=None):
    fmt = fmt or {}
    d = df.copy()
    for c in d.columns:
        if c in fmt: d[c] = d[c].map(fmt[c])
        elif d[c].dtype.kind == "f": d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:.4g}")
    cols = list(d.columns)
    U = {"‖p−p*‖": "$\\|p-p^\\star\\|$", "κ": "$\\kappa$", "σ": "$\\sigma$", "−": "$-$", "∂v̇/∂v": "$\\partial\\dot v/\\partial v$",
         "cos∠(g_κ, g_100)": "$\\cos\\angle(g_\\kappa,g_{100})$", "|J−J_ref|": "$|J-J_{\\rm ref}|$", "≥": "$\\ge$", "→": "$\\to$", "×": "$\\times$", "≈": "$\\approx$"}
    U.update({"∠": "$\\angle$", "‖": "$\\|$", "∂": "$\\partial$", "v̇": "$\\dot v$", "γ": "$\\gamma$", "Δ": "$\\Delta$", "η": "$\\eta$"})
    def T(x):
        x = str(x)
        for a in sorted(U, key=len, reverse=True): x = x.replace(a, U[a])
        return x.replace("_", "\\_").replace("%", "\\%").replace("#", "\\#")
    tex = ["\\begin{tabular}{" + (colspec or ("l" + "r"*(len(cols)-1))) + "}", "\\toprule", " & ".join(T(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in d.iterrows(): tex.append(" & ".join(T(v) for v in r.values) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    if len(cols) > 7: tex = ["\\resizebox{\\textwidth}{!}{"] + tex + ["}"]
    (HERE / f"{name}.tex").write_text("\n".join(tex) + "\n")
    md = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|"*len(cols)]
    md += ["| " + " | ".join(map(str, r.values)) + " |" for _, r in d.iterrows()]
    (HERE / f"{name}.md").write_text("\n".join(md) + "\n")
    print("table", name, d.shape)

# --- T1: headline sweep (main arms), seed medians + paired difference -------------
sw = rd("sweeps.csv"); m = sw[sw.exp == "main"]
piv = m.pivot_table(index="window_size", columns="arm", values=["J", "p_err", "score"], aggfunc="median")
pd_ = m.pivot_table(index=["seed", "window_size"], columns="arm", values="p_err").reset_index()
rows = []
rng = np.random.default_rng(0)
for k, g in pd_.groupby("window_size"):
    x = (g.propagate - g.reset).values; bs = rng.choice(x, size=(2000, len(x))).mean(axis=1)
    rows.append(dict(window_size=k, J_GP=piv.loc[k, ("J", "propagate")], J_best=piv.loc[k, ("J", "best")], J_control=piv.loc[k, ("J", "reset")],
                     perr_GP=piv.loc[k, ("p_err", "propagate")], perr_best=piv.loc[k, ("p_err", "best")], perr_control=piv.loc[k, ("p_err", "reset")],
                     score_GP=piv.loc[k, ("score", "propagate")], score_control=piv.loc[k, ("score", "reset")],
                     paired_diff=x.mean(), ci_lo=np.quantile(bs, 0.025), ci_hi=np.quantile(bs, 0.975), n_GP_better=int((g.propagate < g.reset).sum()), n_plateau_GP=int((g.propagate >= 1.0).sum())))
T1 = pd.DataFrame(rows); T1.to_csv(HERE / "T1_headline.csv", index=False)
write("T1_headline", T1.rename(columns={"window_size": "κ", "J_GP": "J GP", "J_best": "J best-GP", "J_control": "J control", "perr_GP": "‖p−p*‖ GP", "perr_best": "‖p−p*‖ best-GP", "perr_control": "‖p−p*‖ control",
                                        "score_GP": "score GP", "score_control": "score control", "paired_diff": "GP−control", "ci_lo": "CI lo", "ci_hi": "CI hi", "n_GP_better": "#GP better", "n_plateau_GP": "#GP on plateau"}))
NUM["T1_final_perr_GP"] = float(T1.perr_GP.iloc[-1]); NUM["T1_final_perr_control"] = float(T1.perr_control.iloc[-1]); NUM["T1_min_perr_GP"] = float(T1.perr_GP.min()); NUM["T1_argmin_kappa"] = int(T1.window_size[T1.perr_GP.idxmin()])
NUM["T1_final_paired_diff"] = float(T1.paired_diff.iloc[-1]); NUM["T1_final_ci"] = [float(T1.ci_lo.iloc[-1]), float(T1.ci_hi.iloc[-1])]
NUM["seed1_plateau"] = bool((m[(m.arm == "propagate") & (m.seed == 1)].blowup).all()); NUM["n_seeds_main"] = int(m.seed.nunique())
NUM["seed_p_err_range"] = [float(m.seed_p_err.min()), float(m.seed_p_err.max())]
NUM["control_final_blowup_frac"] = float(m[(m.arm == "reset") & (m.window_size == 100)].blowup.mean())

# --- T2: variations at κ=100 (median over seeds) ------------------------------------
def fin(exp, by):
    g = sw[(sw.exp == exp) & (sw.window_size == 100)]
    return g.groupby(by).agg(n_seeds=("seed", "nunique"), p_err=("p_err", "median"), score=("score", "median"), blowup_frac=("blowup", "mean"), wall_total_s=("wall_s", "median")).reset_index()
parts = []
for exp, by in (("schedule", "arm"), ("penalty", "arm"), ("noise", ["arm", "noise"]), ("sparsity", "gamma"), ("optimizer", "arm"), ("iters", "iters"), ("substeps", "S"), ("seedscale", "arm")):
    if (sw.exp == exp).any():
        t = fin(exp, by); t.insert(0, "experiment", exp)
        if isinstance(by, list): t["arm"] = t["arm"] + " σ=" + t["noise"].astype(str); t = t.drop(columns="noise")
        t = t.rename(columns={by if isinstance(by, str) else "arm": "arm"}); t["arm"] = t["arm"].astype(str); parts.append(t)
T2 = pd.concat(parts, ignore_index=True); T2.to_csv(HERE / "T2_variations.csv", index=False)
write("T2_variations", T2.rename(columns={"p_err": "‖p−p*‖ (κ=100)", "score": "score", "blowup_frac": "plateau frac", "wall_total_s": "wall s per κ"}))
# per-exp headline numbers
for exp in T2.experiment.unique():
    g = T2[T2.experiment == exp]; NUM[f"T2_{exp}"] = {str(a): [float(p), float(s)] for a, p, s in zip(g.arm, g.p_err, g.score)}

# --- T3: solvers at κ=5, p* -----------------------------------------------------------
if (RES / "solvers_eval.csv").exists():
    se = rd("solvers_eval.csv"); pt = se.param.unique()[0]; k5 = se[(se.param == pt) & (se.window_size == 5)][["method", "setting", "J", "abs_err_vs_reference", "time_ms"]]
    write("T3_solvers", k5.rename(columns={"abs_err_vs_reference": "|J−J_ref|", "time_ms": "ms per eval"}), fmt={"abs_err_vs_reference": lambda v: f"{v:.2e}", "J": lambda v: f"{v:.7f}", "time_ms": lambda v: f"{v:.3f}"})
    NUM["T3_inhouse_S10_err"] = float(k5[k5.setting.astype(str).str.contains("S=10")].abs_err_vs_reference.iloc[0]); NUM["T3_inhouse_S10_ms"] = float(k5[k5.setting.astype(str).str.contains("S=10")].time_ms.iloc[0])

# --- T4: bugs -------------------------------------------------------------------------
if (RES / "bug_summary.json").exists():
    b = js("bug_summary.json"); f2 = rd("bug_F2_w_residual.csv"); f2t = f2[f2.param == "ptrue"]
    rows = [dict(finding="F2 (R001): mod 7 stiff branch drops the w-residual", measure="share of J(p*) missing, κ=1/10/25/100",
                 value=" / ".join(f"{v*100:.1f} %" for v in f2t.sort_values("window_size").missing_fraction)),
            dict(finding="F1 (R001): mod 6 odefun_new = linear truncation", measure="∂v̇/∂v at u=(1.5,0.4): pre-fix vs fixed", value=f"{b.get('F1_J11_prefix_at_uref', np.nan):.3g} vs {b.get('F1_J11_fixed_at_uref', np.nan):.3g}"),
            dict(finding="F4 (R001): sweep gradient = single-shooting gradient", measure="cos∠(gκ, g100) at κ=1 / 5", value=f"{rd('bug_F4_gradient.csv').set_index('window_size').cosine_similarity.loc[1]:.3f} / {b.get('F4_cosine_similarity_k5', np.nan):.3f}"),
            dict(finding="NEW: mod 8 driver mixed monomial orders", measure="spurious ‖p−p*‖ between orderings", value=f"{b.get('ordering_spurious_p_err', np.nan):.3f}"),
            dict(finding="NEW: mod 8 forward_simulation_loss ignores data_", measure="J identical for 3 datasets pre-fix / differs post-fix", value=f"{b.get('shadowing_prefix_identical', '?')} / {b.get('shadowing_fixed_differs', '?')}")]
    write("T4_bugs", pd.DataFrame(rows), colspec="p{0.36\\textwidth} p{0.33\\textwidth} p{0.24\\textwidth}"); NUM["T4"] = rows

# --- T5: other systems ----------------------------------------------------------------
if (RES / "other_sweeps.csv").exists():
    o = rd("other_sweeps.csv"); o100 = o[o.window_size == 100]
    T5 = o100.groupby(["system", "optimizer", "arm"]).agg(n_seeds=("seed", "nunique"), p_err=("p_err", "median"), score=("score", "median"), blowup_frac=("blowup", "mean")).reset_index()
    write("T5_other_systems", T5.rename(columns={"p_err": "‖p−p*‖ (κ=100)", "blowup_frac": "plateau frac"}))
    NUM["T5"] = {f"{s}/{op}/{a}": [float(p), float(sc)] for s, op, a, p, sc in zip(T5.system, T5.optimizer, T5.arm, T5.p_err, T5.score)}

# --- protocol constants + misc -------------------------------------------------------
meta = js("fhn_data_meta.json"); cm = js("concept_meta.json"); cc = js("core_check.json")
NUM.update({"N": meta["N"], "Delta_t": meta["Delta_t"], "noise_rel": meta["noise_rel"], "eta_max": meta["eta_max_norm"], "noise_floor_J": meta["noise_floor_J"],
            "L_traj": cm["L_traj"], "L_box": cm["L_box"], "Ltilde": cm["Ltilde_traj"], "lognorm": cm["lognorm_traj"],
            "core_loss_reldiff": cc["loss_max_reldiff"], "core_data_absdiff": cc["dataset_max_absdiff"], "ms_per_eval": cc["ms_per_loss_eval_core_k5"]})
h = rd("hessian_at_ptrue.csv"); NUM["hess_lambda_max_k1"] = float(h.lambda_max.iloc[0]); NUM["hess_lambda_max_k100"] = float(h.lambda_max.iloc[-1]); NUM["hess_nneg_k100"] = int(h.n_negative.iloc[-1])
mn = rd("landscape_1d_minima.csv").groupby("window_size").n_local_minima.mean(); NUM["minima_k1"] = float(mn.loc[1]); NUM["minima_k100"] = float(mn.loc[100])
(HERE / "table_numbers.json").write_text(json.dumps(NUM, indent=1, default=float))
print("numbers:", len(NUM))
