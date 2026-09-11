"""make_tables.py -- build tables/ for R004 from external_data/ (inherited R002/R003 outputs).
Copies the inherited LaTeX/Markdown tables verbatim, derives the compact variations table from
T2_variations.csv, and writes tables/numbers.tex (every number the manuscript quotes in prose)
plus tables/numbers.json for the gates.  Nothing here re-does any numerics of R002/R003.
"""
from pathlib import Path
import json, shutil
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; RPT = HERE.parent; EXT = RPT / "external_data"; TAB = RPT / "tables"; TAB.mkdir(exist_ok=True)
# 1. verbatim copies
for f in ("T1_headline.tex", "T1_headline.md", "T5_other_systems.tex", "T5_other_systems.md", "lemmas.tex", "windows.tex", "constants.tex"):
    shutil.copyfile(EXT / "tables" / f, TAB / f)
# markdown twins of the R003 tables (values inline)
(TAB / "lemmas.md").write_text("""| $t=\\Delta T$ | Lemma 1 original $e^{L\\Delta T}$ | Lemma 1 proposed $e^{\\mu\\Delta T}$ | Lemma 1 observed peak | Lemma 2 original $\\frac{\\tilde L}{L}(e^{L\\Delta T}-1)$ | Lemma 2 proposed $\\frac{\\tilde L}{\\mu}(e^{\\mu\\Delta T}-1)$ | Lemma 2 observed peak |
|---|---|---|---|---|---|---|
| 1 | 21.1 | 3.23 | 2.93 | 69.8 | 20.1 | 3.7 |
| 2 | 447 | 10.4 | 6.92 | $1.5\\times10^{3}$ | 84.9 | 6.7 |
| 5 | $4.2\\times10^{6}$ | 349 | 8.48 | $1.5\\times10^{7}$ | $3.1\\times10^{3}$ | 12.0 |
| 10 | $1.8\\times10^{13}$ | $1.2\\times10^{5}$ | 8.48 | $6.2\\times10^{13}$ | $1.1\\times10^{6}$ | 15.1 |
""")
(TAB / "windows.md").write_text("""| $\\Delta T$ | windows | a priori $e^{\\mu\\Delta T}$ | Euclidean $e^{\\Lambda_k}$ worst | Euclidean median | weighted $e^{\\Lambda^D_k}$ worst | weighted median | contracting windows |
|---|---|---|---|---|---|---|---|
| 1 | 100 | 3.23 | 3.10 | 1.16 | 2.60 | 0.94 | 74 % |
| 2 | 50 | 10.4 | 8.56 | 1.41 | 5.89 | 0.88 | 70 % |
| 5 | 20 | 349 | 34.78 | 3.75 | 8.04 | 0.73 | 55 % |
| 10 | 10 | $1.2\\times10^{5}$ | 138.79 | 29.25 | 5.89 | 3.20 | 30 % |
""")
# 2. compact variations table (selected rows of T2)
t2 = pd.read_csv(EXT / "tables" / "T2_variations.csv")
keep = [("schedule", "dense", "schedule DENSE (44 stages)"), ("schedule", "coarse", "schedule COARSE (4 stages)"), ("schedule", "jump", "schedule JUMP (2 stages)"),
        ("penalty", "propagate_graded", "graded blow-up penalty, GP"), ("penalty", "reset_graded", "graded blow-up penalty, control"),
        ("noise", "propagate σ=0.0", "noise 0 %, GP"), ("noise", "propagate σ=0.1", "noise 10 %, GP"), ("noise", "propagate σ=0.2", "noise 20 %, GP"), ("noise", "reset σ=0.0", "noise 0 %, control"),
        ("sparsity", "0.0", "$\\gamma=0$, GP"), ("sparsity", "0.05", "$\\gamma=0.05$, GP"), ("sparsity", "1.0", "$\\gamma=1$, GP"),
        ("optimizer", "bfgs_propagate", "BFGS, GP"), ("optimizer", "bfgs_reset", "BFGS, control"),
        ("iters", "1000", "Nelder--Mead 1000 iterations, GP"), ("iters", "5000", "Nelder--Mead 5000 iterations, GP"),
        ("substeps", "1", "integrator sub-steps $S=1$, GP"), ("substeps", "5", "integrator sub-steps $S=5$, GP")]
rows_tex, rows_md = [], []
for exp, arm, label in keep:
    r = t2[(t2.experiment == exp) & (t2.arm.astype(str) == arm)].iloc[0]
    rows_tex.append(f"{label.replace(chr(32)+chr(37), chr(92)+chr(44)+chr(92)+chr(37))} & {int(r.n_seeds)} & {r.p_err:.2f} & {r.score:.2f} & {100*r.blowup_frac:.0f}\\,\\% \\\\")
    rows_md.append(f"| {label.replace('--','–')} | {int(r.n_seeds)} | {r.p_err:.2f} | {r.score:.2f} | {100*r.blowup_frac:.0f} % |")
(TAB / "T2_compact.tex").write_text("\\begin{tabular}{lrrrr}\n\\toprule\nvariation (arm) & seeds & $\\|p-p^\\star\\|$ at $\\kappa=100$ & score & seeds on plateau\\\\\n\\midrule\n" + "\n".join(rows_tex) + "\n\\bottomrule\n\\end{tabular}\n")
(TAB / "T2_compact.md").write_text("| variation (arm) | seeds | $\\|p-p^\\star\\|$ at $\\kappa=100$ | score | seeds on plateau |\n|---|---|---|---|---|\n" + "\n".join(rows_md) + "\n")
# 3. numbers quoted in prose
TN = json.loads((EXT / "tables" / "table_numbers.json").read_text()); S3 = json.loads((EXT / "tables" / "R003_summary.json").read_text())
CM = json.loads((EXT / "concept_meta.json").read_text()); DM = json.loads((EXT / "fhn_data_meta.json").read_text())
mn = pd.read_csv(EXT / "landscape_1d_minima.csv"); hs = pd.read_csv(EXT / "hessian_at_ptrue.csv"); ph = pd.read_csv(EXT / "post_hessian_at_minimizers.csv"); cc = pd.read_csv(EXT / "concept_costs.csv")
ph7 = ph[ph.seed != 1]  # plateau seed excluded, as in R002 Figure 37
N = dict(
    perrGPfinal=TN["T1_final_perr_GP"], perrCtrlFinal=TN["T1_final_perr_control"], perrGPmin=TN["T1_min_perr_GP"], argminKappa=TN["T1_argmin_kappa"],
    pairedDiff=TN["T1_final_paired_diff"], ciLo=TN["T1_final_ci"][0], ciHi=TN["T1_final_ci"][1], ctrlBlowupFrac=TN["control_final_blowup_frac"],
    seedErrLo=TN["seed_p_err_range"][0], seedErrHi=TN["seed_p_err_range"][1], nSeeds=TN["n_seeds_main"],
    schedDense=TN["T2_schedule"]["dense"][0], schedCoarse=TN["T2_schedule"]["coarse"][0], schedJump=TN["T2_schedule"]["jump"][0],
    gradedGP=TN["T2_penalty"]["propagate_graded"][0], noiseTenGP=TN["T2_noise"]["propagate σ=0.1"][0], noiseTwentyGP=TN["T2_noise"]["propagate σ=0.2"][0],
    bfgsGP=TN["T2_optimizer"]["bfgs_propagate"][0], bfgsGPscore=TN["T2_optimizer"]["bfgs_propagate"][1],
    minimaKone=float(mn[mn.window_size == 1].n_local_minima.mean()), minimaKhundred=float(mn[mn.window_size == 100].n_local_minima.mean()),
    lamMaxKone=float(hs[hs.window_size == 1].lambda_max.iloc[0]), lamMaxKhundred=float(hs[hs.window_size == 100].lambda_max.iloc[0]),
    gradKone=float(hs[hs.window_size == 1].grad_norm.iloc[0]), gradKhundred=float(hs[hs.window_size == 100].grad_norm.iloc[0]),
    nNegMax=int(hs.n_negative.max()), fracPD=float((ph7.n_negative == 0).mean()),
    JtrueHundred=float(cc[(cc.param == "true") & (cc.window_size == 100)].J.iloc[0]), JwrongHundred=float(cc[(cc.param == "wrong") & (cc.window_size == 100)].J.iloc[0]),
    JwrongTen=float(cc[(cc.param == "wrong") & (cc.window_size == 10)].J.iloc[0]), pWrongErr=CM["p_wrong_err"],
    L=CM["L_traj"], Lbox=CM["L_box"], mu=CM["lognorm_traj"], Ltilde=CM["Ltilde_traj"], etaMax=DM["eta_max_norm"], Nobs=DM["N"], noiseFloor=DM["noise_floor_J"],
    peakAmp=S3["lemma1_peak_amplification"], peakSens=S3["lemma2_peak_sensitivity"], weightS=S3["weight_s"], fracDneg=S3["frac_time_lognorm_D_negative"],
    attractorOverNoise=S3["attractor_over_noise"], noiseSlack=S3["noise_slack_factor"], period=S3["orbit_period"], muLower=S3["lognorm_lower_fine"],
)
# T5 numbers from the markdown twin (same content as the .tex)
t5 = [l for l in (EXT / "tables" / "T5_other_systems.md").read_text().splitlines() if l.startswith("| l")]
def t5row(sys_, opt, arm):
    for l in t5:
        c = [x.strip() for x in l.strip("|").split("|")]
        if c[0] == sys_ and c[1] == opt and c[2] == arm: return float(c[4]), float(c[5]), float(c[6])
N.update(dict(lvGP=t5row("lv", "nm", "propagate")[0], lvGPscore=t5row("lv", "nm", "propagate")[1], lvCtrl=t5row("lv", "nm", "reset")[0], lvCtrlPlateau=t5row("lv", "nm", "reset")[2],
              lorenzBFGS=t5row("lorenz", "bfgs", "propagate")[0], lorenzBFGSscore=t5row("lorenz", "bfgs", "propagate")[1], lorenzNM=t5row("lorenz", "nm", "propagate")[0], lorenzNMscore=t5row("lorenz", "nm", "propagate")[1]))
(TAB / "numbers.json").write_text(json.dumps(N, indent=1))
fmt = dict(perrGPfinal="{:.2f}", perrCtrlFinal="{:.2f}", perrGPmin="{:.2f}", argminKappa="{:d}", pairedDiff="{:.2f}", ciLo="{:.2f}", ciHi="{:.2f}", ctrlBlowupFrac="{:.0%}",
           seedErrLo="{:.2f}", seedErrHi="{:.2f}", nSeeds="{:d}", schedDense="{:.2f}", schedCoarse="{:.2f}", schedJump="{:.2f}", gradedGP="{:.2f}", noiseTenGP="{:.2f}", noiseTwentyGP="{:.2f}",
           bfgsGP="{:.2f}", bfgsGPscore="{:.2f}", minimaKone="{:.2f}", minimaKhundred="{:.1f}", lamMaxKone="{:.0f}", lamMaxKhundred="{:.1e}", gradKone="{:.2f}", gradKhundred="{:.0f}", nNegMax="{:d}", fracPD="{:.0%}",
           JtrueHundred="{:.4f}", JwrongHundred="{:.2f}", JwrongTen="{:.4f}", pWrongErr="{:.3f}", L="{:.2f}", Lbox="{:.2f}", mu="{:.2f}", Ltilde="{:.2f}", etaMax="{:.3f}", Nobs="{:d}", noiseFloor="{:.4f}",
           peakAmp="{:.1f}", peakSens="{:.1f}", weightS="{:.2f}", fracDneg="{:.0%}", attractorOverNoise="{:.0f}", noiseSlack="{:.1f}", period="{:.1f}", muLower="{:.2f}",
           lvGP="{:.2f}", lvGPscore="{:.2f}", lvCtrl="{:.2f}", lvCtrlPlateau="{:.0%}", lorenzBFGS="{:.1f}", lorenzBFGSscore="{:.2f}", lorenzNM="{:.1f}", lorenzNMscore="{:.2f}")
out = []
for k, v in N.items():
    s = fmt[k].format(v).replace("%", "\\%")
    if "e" in s and k == "lamMaxKhundred":
        m, e = s.split("e"); s = f"\\ensuremath{{{m}\\times10^{{{int(e)}}}}}"
    out.append(f"\\newcommand{{\\n{k}}}{{{s}}}")
(TAB / "numbers.tex").write_text("\n".join(out) + "\n")
print((TAB / "numbers.tex").read_text()); print((TAB / "T2_compact.tex").read_text())
