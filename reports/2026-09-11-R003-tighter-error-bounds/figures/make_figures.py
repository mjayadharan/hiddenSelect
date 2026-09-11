"""make_figures.py -- the single figure of R003 (reads analysis/results/, writes figures/fig01_bounds.{pdf,png})."""
from pathlib import Path
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent; RES = HERE.parent / "analysis" / "results"
env = pd.read_csv(RES / "envelopes.csv"); loc = pd.read_csv(RES / "local_lognorm.csv"); locD = pd.read_csv(RES / "local_lognorm_D.csv")
S = json.loads((RES / "summary.json").read_text())
BLUE, VERM, GREEN, GREY, PURPLE = "#0072B2", "#D55E00", "#009E73", "#555555", "#7B4EA3"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#dddddd", "grid.linewidth": 0.5, "legend.frameon": False})
fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.2))
t = env.t
a = ax[0]
a.semilogy(t, env.lipschitz_l1, color=VERM, lw=2, label=r"Lipschitz $e^{Lt}$, $L=%.2f$" % S["L_traj"])
a.semilogy(t, env.lognorm_l1, color=GREEN, lw=2, ls="--", label=r"log-norm $e^{\mu t}$, $\mu=%.2f$" % S["lognorm_traj"])
a.semilogy(t, env.obs_max_l1, color=BLUE, lw=2, label="observed (max of 24 probes)")
a.set_ylim(0.5, 1e14); a.set_xlabel("$t$ (time units)"); a.set_ylabel(r"$\|\varphi(t;x_1)-\varphi(t;x_2)\|/\|x_1-x_2\|$")
a.set_title("(a) Lemma 1: initial-state sensitivity", loc="left"); a.legend(loc="upper left", fontsize=7.5)
a = ax[1]
a.semilogy(t, env.lipschitz_l2, color=VERM, lw=2, label=r"$\frac{\tilde L}{L}(e^{Lt}-1)$")
a.semilogy(t, env.lognorm_l2, color=GREEN, lw=2, ls="--", label=r"$\frac{\tilde L}{\mu}(e^{\mu t}-1)$")
a.semilogy(t, env.obs_max_l2, color=BLUE, lw=2, label="observed (max of 24 probes)")
a.set_ylim(0.1, 1e14); a.set_xlabel("$t$ (time units)"); a.set_ylabel(r"$\|\varphi(t;p_1,x_0)-\varphi(t;p_2,x_0)\|/\|p_1-p_2\|$")
a.set_title("(b) Lemma 2: parameter sensitivity", loc="left"); a.legend(loc="upper left", fontsize=7.5)
a = ax[2]
m = loc.t <= 40
a.plot(loc.t[m], loc.lognorm_local[m], color=GREEN, lw=1.8, label=r"$\mu(x^\star(t))$, Euclidean norm")
a.plot(locD.t[m], locD.lognorm_local_D[m], color=PURPLE, lw=1.8, label=r"$\mu_D(x^\star(t))$, weighted norm $\|Dx\|$")
a.plot(loc.t[m], loc.opnorm_local[m], color=VERM, lw=1.2, ls=":", label=r"$\|\partial f/\partial x\|_2$ (local $L$)")
a.axhline(0, color=GREY, lw=0.8); a.set_xlabel("$t$ (time units, one orbit period is %.1f)" % S["orbit_period"]); a.set_ylabel("local growth rate")
a.set_title("(c) Local rates along the FHN orbit", loc="left"); a.legend(loc="upper right", fontsize=7.5); a.set_ylim(-0.6, 3.4)
fig.tight_layout(w_pad=1.5)
for ext in ("pdf", "png"): fig.savefig(HERE / f"fig01_bounds.{ext}", dpi=160)
print("figure written")
