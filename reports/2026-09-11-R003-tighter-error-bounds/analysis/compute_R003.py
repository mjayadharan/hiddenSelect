"""compute_R003.py -- recompute every number R003 quotes from the copied R002 inputs.

Inputs (external_data/, hashes in INPUTS.md): concept_meta.json, fhn_data_meta.json,
lemma1_flow_sensitivity.csv, lemma2_param_sensitivity.csv, fhn_fine.csv, fhn_data.csv,
fhn_library.csv.  Writes analysis/results/{summary.json, lemma1_table.csv,
lemma2_table.csv, window_exponents.csv, local_lognorm.csv, envelopes.csv}.
No package beyond numpy/pandas.  Run from anywhere.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
EXT = HERE.parent / "external_data"
RES = HERE / "results"; RES.mkdir(exist_ok=True)

meta = json.loads((EXT / "concept_meta.json").read_text())
dmeta = json.loads((EXT / "fhn_data_meta.json").read_text())
lib = pd.read_csv(EXT / "fhn_library.csv")
fine = pd.read_csv(EXT / "fhn_fine.csv")
data = pd.read_csv(EXT / "fhn_data.csv")
l1 = pd.read_csv(EXT / "lemma1_flow_sensitivity.csv")
l2 = pd.read_csv(EXT / "lemma2_param_sensitivity.csv")

# ---- FHN right-hand side and Jacobian built from the library file (no hard-coded model) ----
def parse_mono(s):
    ev = ew = 0
    for tok in str(s).split("*"):
        tok = tok.strip()
        if tok == "1": continue
        m = re.fullmatch(r"([vw])(?:\^(\d+))?", tok); var, e = m.group(1), int(m.group(2) or 1)
        if var == "v": ev += e
        else: ew += e
    return ev, ew
terms = [(r.equation, *parse_mono(r.monomial), r.p_true) for r in lib.itertuples() if r.p_true != 0.0]
def jac(v, w):
    J = np.zeros((2, 2))
    for eq, ev, ew, c in terms:
        i = 0 if eq == "dv/dt" else 1
        if ev: J[i, 0] += c * ev * v**(ev-1) * w**ew
        if ew: J[i, 1] += c * ew * v**ev * w**(ew-1)
    return J
lognorm = lambda J: np.linalg.eigvalsh((J + J.T) / 2)[-1]
lognorm_lo = lambda J: np.linalg.eigvalsh((J + J.T) / 2)[0]
opn = lambda J: np.linalg.norm(J, 2)

# ---- constants along the fine orbit and on the R002 box ---------------------------------
Js = [jac(v, w) for v, w in zip(fine.v, fine.w)]
L_fine = max(opn(J) for J in Js); mu_fine = max(lognorm(J) for J in Js); mu_lo_fine = min(lognorm_lo(J) for J in Js)
box = [jac(v, w) for v in np.linspace(-2.5, 2.5, 41) for w in np.linspace(-1, 2, 41)]
L_box = max(opn(J) for J in box); mu_box = max(lognorm(J) for J in box)
L, mu, Lt = meta["L_traj"], meta["lognorm_traj"], meta["Ltilde_traj"]
# analytic global sup of the FHN log norm: 1 - v^2 <= 1 attained at v = 0
J0 = jac(0.0, float(fine.w.iloc[0])); mu_global_analytic = lognorm(J0)

# ---- Lemma 1 / Lemma 2 envelopes over the 24 probes ------------------------------------
env1 = l1.groupby("t").ratio.max().rename("obs_max").reset_index()
env1["lipschitz"] = np.exp(L * env1.t); env1["lognorm"] = np.exp(mu * env1.t)
env2 = l2.groupby("t").ratio.max().rename("obs_max").reset_index()
env2["lipschitz"] = Lt / L * (np.exp(L * env2.t) - 1); env2["lognorm"] = Lt / mu * (np.exp(mu * env2.t) - 1)
env = env1.merge(env2, on="t", suffixes=("_l1", "_l2")); env.to_csv(RES / "envelopes.csv", index=False)

rows = []
for dT in (1, 2, 5, 10):
    sub = l1[l1.t <= dT + 1e-9]
    rows.append(dict(DeltaT=dT, lipschitz=np.exp(L*dT), lognorm=np.exp(mu*dT), observed_peak=sub.ratio.max(),
                     ratio_lip_over_obs=np.exp(L*dT)/sub.ratio.max(), ratio_log_over_obs=np.exp(mu*dT)/sub.ratio.max()))
t1 = pd.DataFrame(rows); t1.to_csv(RES / "lemma1_table.csv", index=False)
rows = []
for dT in (1, 2, 5, 10):
    sub = l2[l2.t <= dT + 1e-9]
    rows.append(dict(DeltaT=dT, lipschitz=Lt/L*(np.exp(L*dT)-1), lognorm=Lt/mu*(np.exp(mu*dT)-1), observed_peak=sub.ratio.max()))
t2 = pd.DataFrame(rows); t2.to_csv(RES / "lemma2_table.csv", index=False)

# gate material: do the bounds hold probe by probe?
tol = 1e-9
l1_lip_ok = bool((l1.ratio <= np.exp(L*l1.t) + tol).all()); l1_log_ok = bool((l1.ratio <= np.exp(mu*l1.t) + tol).all())
l1_lo_ok = bool((l1.ratio >= np.exp(-L*l1.t) - tol).all()); l1_lo_log_ok = bool((l1.ratio >= np.exp(mu_lo_fine*l1.t) - tol).all())
l2_lip_ok = bool((l2.ratio <= Lt/L*(np.exp(L*l2.t)-1) + tol).all()); l2_log_ok = bool((l2.ratio <= Lt/mu*(np.exp(mu*l2.t)-1) + tol).all())
l1_max_viol_log = float((l1.ratio / np.exp(mu*l1.t)).max())

# ---- local log norm along the orbit and per-window integrated exponents -----------------
t = fine.t.to_numpy(); mus = np.array([lognorm(J) for J in Js]); Lloc = np.array([opn(J) for J in Js])
pd.DataFrame(dict(t=t, v=fine.v, w=fine.w, lognorm_local=mus, opnorm_local=Lloc)).to_csv(RES / "local_lognorm.csv", index=False)
dt = float(np.median(np.diff(t)))
frac_neg = float((mus < 0).mean()); mean_mu = float(np.trapz(mus, t) / (t[-1]-t[0]))
# period from successive upward crossings of v = 0
cross = np.where((fine.v.to_numpy()[:-1] < 0) & (fine.v.to_numpy()[1:] >= 0))[0]
period = float(np.mean(np.diff(t[cross]))) if len(cross) > 1 else float("nan")
wrows = []
for dT in (1, 2, 5, 10):
    n = int(round(dT / dt)); starts = range(0, len(t) - n, n)
    Lam = np.array([np.trapz(mus[s:s+n+1], t[s:s+n+1]) for s in starts])
    wrows.append(dict(DeltaT=dT, n_windows=len(Lam), exp_Lambda_max=float(np.exp(Lam.max())), exp_Lambda_median=float(np.exp(np.median(Lam))),
                      exp_Lambda_min=float(np.exp(Lam.min())), frac_windows_contracting=float((Lam < 0).mean()),
                      lognorm_bound=float(np.exp(mu*dT)), lipschitz_bound=float(np.exp(L*dT))))
wt = pd.DataFrame(wrows); wt.to_csv(RES / "window_exponents.csv", index=False)

# ---- noise-slack and attractor-size slack --------------------------------------------------
eta = data[["v", "w"]].to_numpy() - data[["v_clean", "w_clean"]].to_numpy()
eta_sq = (eta**2).sum(1); eta_max = float(np.sqrt(eta_sq.max())); eta_mean_sq = float(eta_sq.mean())
y_norm_max = float(np.sqrt((data[["v", "w"]].to_numpy()**2).sum(1)).max())
d, N = 2, len(data)

summary = dict(
    source_commit="cafad1ac6c4894334c51ebe52291d3f114abeefb (R002 result files copied to external_data/)",
    L_traj=L, L_box_R002=meta["L_box"], lognorm_traj=mu, Ltilde_traj=Lt,
    L_fine_recomputed=L_fine, L_box_recomputed=L_box, lognorm_fine_recomputed=mu_fine, lognorm_box_recomputed=mu_box,
    lognorm_lower_fine=mu_lo_fine, lognorm_global_analytic=float(mu_global_analytic),
    eta_max_norm=eta_max, eta_max_norm_R002=dmeta["eta_max_norm"], eta_mean_sq=eta_mean_sq,
    noise_slack_factor=eta_max**2 / eta_mean_sq, noise_slack_gaussian_rule=1 + 2*np.log(N)/d,
    y_norm_max=y_norm_max, attractor_over_noise=y_norm_max / eta_max,
    N=N, d=d, orbit_period=period, frac_time_lognorm_negative=frac_neg, mean_lognorm_over_record=mean_mu,
    lemma1_peak_amplification=float(l1.ratio.max()), lemma2_peak_sensitivity=float(l2.ratio.max()),
    l1_lipschitz_upper_holds=l1_lip_ok, l1_lognorm_upper_holds=l1_log_ok, l1_lipschitz_lower_holds=l1_lo_ok,
    l1_lognorm_lower_holds=l1_lo_log_ok, l1_max_ratio_to_lognorm_bound=l1_max_viol_log,
    l2_lipschitz_holds=l2_lip_ok, l2_lognorm_holds=l2_log_ok,
    lemma2_saturation_if_mu_neg="Ltilde/|mu| -- not applicable here since mu = %.3f > 0" % mu,
)
(RES / "summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2)); print(t1.round(3)); print(t2.round(3)); print(wt.round(3))

# ---- weighted (diagonal) norm: ||x||_D = ||D x||, D = diag(1, s) --------------------------
# J_D = D J D^{-1}: the off-diagonals become -1/s and 0.08 s; s = sqrt(|J12|/|J21|) makes the
# symmetric part diagonal, so mu_D(x) = max(1 - v^2, -0.064) locally.
J12 = abs(jac(0.0, 0.0)[0, 1]); J21 = abs(jac(0.0, 0.0)[1, 0]); s = float(np.sqrt(J12 / J21))
D = np.diag([1.0, s]); Dinv = np.diag([1.0, 1/s])
musD = np.array([lognorm(D @ J @ Dinv) for J in Js])
muD_global = float(max(lognorm(D @ J @ Dinv) for J in box))
pd.DataFrame(dict(t=t, lognorm_local_D=musD)).to_csv(RES / "local_lognorm_D.csv", index=False)
wrowsD = []
for dT in (1, 2, 5, 10):
    n = int(round(dT / dt)); starts = range(0, len(t) - n, n)
    Lam = np.array([np.trapz(musD[s_:s_+n+1], t[s_:s_+n+1]) for s_ in starts])
    wrowsD.append(dict(DeltaT=dT, n_windows=len(Lam), exp_Lambda_max=float(np.exp(Lam.max())), exp_Lambda_median=float(np.exp(np.median(Lam))),
                       exp_Lambda_min=float(np.exp(Lam.min())), frac_windows_contracting=float((Lam < 0).mean()),
                       lognorm_D_bound=float(np.exp(muD_global*dT)), cond_D=s))
wtD = pd.DataFrame(wrowsD); wtD.to_csv(RES / "window_exponents_D.csv", index=False)
summary.update(dict(weight_s=s, lognorm_D_global=muD_global, frac_time_lognorm_D_negative=float((musD < 0).mean()),
                    mean_lognorm_D_over_record=float(np.trapz(musD, t) / (t[-1]-t[0]))))
(RES / "summary.json").write_text(json.dumps(summary, indent=2))
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print(wt.round(3)); print(wtD.round(3)); print({k: summary[k] for k in ("weight_s","lognorm_D_global","frac_time_lognorm_D_negative","mean_lognorm_D_over_record")})
