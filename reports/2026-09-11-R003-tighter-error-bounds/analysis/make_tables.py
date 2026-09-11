"""make_tables.py -- LaTeX tables for R003 from analysis/results/*.csv (numbers never typed by hand)."""
from pathlib import Path
import json, pandas as pd
HERE = Path(__file__).resolve().parent; RES = HERE / "results"; TAB = HERE.parent / "tables"; TAB.mkdir(exist_ok=True)
S = json.loads((RES / "summary.json").read_text())
def sci(x):
    if x < 1000: return f"{x:.3g}" if x < 100 else f"{x:.0f}"
    m, e = f"{x:.1e}".split("e"); return rf"${m}\times10^{{{int(e)}}}$"
t1 = pd.read_csv(RES / "lemma1_table.csv"); t2 = pd.read_csv(RES / "lemma2_table.csv")
w = pd.read_csv(RES / "window_exponents.csv"); wD = pd.read_csv(RES / "window_exponents_D.csv")
rows = []
for a, b in zip(t1.itertuples(), t2.itertuples()):
    rows.append(f"{int(a.DeltaT)} & {sci(a.lipschitz)} & {sci(a.lognorm)} & {a.observed_peak:.2f} & {sci(b.lipschitz)} & {sci(b.lognorm)} & {b.observed_peak:.1f} \\\\")
(TAB / "lemmas.tex").write_text(
r"""\begin{tabular}{r rrr rrr}
\toprule
& \multicolumn{3}{c}{Lemma 1: $\lVert\varphi(t;x_1)-\varphi(t;x_2)\rVert/\lVert x_1-x_2\rVert$} & \multicolumn{3}{c}{Lemma 2: $\lVert\varphi(t;p_1,x_0)-\varphi(t;p_2,x_0)\rVert/\lVert p_1-p_2\rVert$}\\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}
$t=\Delta T$ & original $e^{L\Delta T}$ & proposed $e^{\mu\Delta T}$ & observed peak & original $\frac{\tilde L}{L}(e^{L\Delta T}-1)$ & proposed $\frac{\tilde L}{\mu}(e^{\mu\Delta T}-1)$ & observed peak\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
rows = []
for a, b in zip(w.itertuples(), wD.itertuples()):
    rows.append(f"{int(a.DeltaT)} & {int(a.n_windows)} & {sci(a.lognorm_bound)} & {a.exp_Lambda_max:.2f} & {a.exp_Lambda_median:.2f} & {b.exp_Lambda_max:.2f} & {b.exp_Lambda_median:.2f} & {100*b.frac_windows_contracting:.0f}\\,\\% \\\\")
(TAB / "windows.tex").write_text(
r"""\begin{tabular}{r r r rr rr r}
\toprule
& & a priori & \multicolumn{2}{c}{per-window $e^{\Lambda_k}$, Euclidean} & \multicolumn{3}{c}{per-window $e^{\Lambda^D_k}$, weighted norm}\\
\cmidrule(lr){4-5}\cmidrule(lr){6-8}
$\Delta T$ & windows & $e^{\mu\Delta T}$ & worst & median & worst & median & contracting windows\\
\midrule
""" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
# constants macro file
M = {"Ltraj": f"{S['L_traj']:.2f}", "Lbox": f"{S['L_box_R002']:.2f}", "mu": f"{S['lognorm_traj']:.2f}", "Ltilde": f"{S['Ltilde_traj']:.2f}",
     "etamax": f"{S['eta_max_norm']:.3f}", "Nobs": str(S['N']), "noiseslack": f"{S['noise_slack_factor']:.1f}", "noiseslackrule": f"{S['noise_slack_gaussian_rule']:.1f}",
     "ynormmax": f"{S['y_norm_max']:.2f}", "attractoroverone": f"{S['attractor_over_noise']:.0f}", "peakamp": f"{S['lemma1_peak_amplification']:.1f}",
     "peaksens": f"{S['lemma2_peak_sensitivity']:.1f}", "period": f"{S['orbit_period']:.1f}", "meanmu": f"{S['mean_lognorm_over_record']:.2f}",
     "weights": f"{S['weight_s']:.2f}", "muDglobal": f"{S['lognorm_D_global']:.2f}", "fracDneg": f"{100*S['frac_time_lognorm_D_negative']:.0f}",
     "meanmuD": f"{S['mean_lognorm_D_over_record']:.3f}", "mulower": f"{S['lognorm_lower_fine']:.2f}"}
(TAB / "constants.tex").write_text("".join(f"\\newcommand{{\\c{k}}}{{{v}}}\n" for k, v in M.items()))
print(open(TAB / "lemmas.tex").read()); print(open(TAB / "windows.tex").read()); print(open(TAB / "constants.tex").read())
