"""make_figures.py — every RESULT figure of R002 (concept schematics live in concept_figures.py).

Usage: python make_figures.py [fig_name ...]   (no args = all). Each figure is a function
registered in FIGS; a figure whose inputs are missing is skipped with a message so the
script never fabricates a panel. All numbers come from analysis/results/*.csv|json.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import *
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIGS = {}
def figure(fn):
    FIGS[fn.__name__] = fn; return fn

def med_iqr(df, by, col):
    g = df.groupby(by)[col]
    return g.median(), g.quantile(0.25), g.quantile(0.75)

def boot_ci(x, B=2000, seed=0):
    """Percentile bootstrap CI of the mean of paired differences x (resampling unit = seed)."""
    rng = np.random.default_rng(seed); x = np.asarray(x, float); n = len(x)
    if n == 0: return np.nan, np.nan, np.nan
    m = rng.choice(x, size=(B, n), replace=True).mean(axis=1)
    return x.mean(), np.quantile(m, 0.025), np.quantile(m, 0.975)

# ------------------------------------------------------------------ theory --------
@figure
def fig08_lemma1_flow_sensitivity():
    d = read_csv("lemma1_flow_sensitivity.csv"); meta = read_json("concept_meta.json")
    L, mu = meta["L_traj"], meta["lognorm_traj"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    ax = axs[0]
    for (s, k), g in d.groupby(["start_index", "direction"]):
        ax.plot(g.t, g.ratio, color=BLUE, alpha=0.25, lw=0.9)
    t = np.linspace(0, 10, 200)
    ax.plot(t, np.exp(L*t), color=VERMILION, lw=1.8, label=r"$e^{L t}$, $L=\sup\|\partial f/\partial x\|_2$")
    ax.plot(t, np.exp(mu*t), color=GREEN, lw=1.8, ls="--", label=r"$e^{\mu t}$, $\mu$ = log-norm")
    ax.plot(t, np.exp(-L*t), color=VERMILION, lw=1.2, ls=":", label=r"$e^{-L t}$")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(r"$\|\varphi(t;x_1)-\varphi(t;x_2)\| / \|x_1-x_2\|$")
    ax.set_title("Lemma 1: sensitivity to the initial state (24 probes)", fontsize=9)
    ax.legend(loc="upper left"); ax.set_ylim(1e-3, 1e6); panel_label(ax, "a")
    ax = axs[1]
    sub = d[d.t <= 3]
    for (s, k), g in sub.groupby(["start_index", "direction"]):
        ax.plot(g.t, g.ratio, color=BLUE, alpha=0.3, lw=0.9)
    t = np.linspace(0, 3, 100)
    ax.plot(t, np.exp(L*t), color=VERMILION, lw=1.8); ax.plot(t, np.exp(mu*t), color=GREEN, lw=1.8, ls="--")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel("amplification of a state perturbation")
    ax.set_title(r"zoom: one window, $\Delta T=\kappa\Delta t$, $\Delta t=1$", fontsize=9)
    for k in (1, 2, 3):
        ax.axvline(k, color=LIGHTGREY, lw=0.8, zorder=0); ax.text(k, 1.15e-1, rf"$\kappa={k}$", ha="center", fontsize=7, color=MUTED)
    ax.set_ylim(1e-1, 1e4); panel_label(ax, "b")
    save(fig, "fig08_lemma1_flow_sensitivity")

@figure
def fig09_lemma2_param_sensitivity():
    d = read_csv("lemma2_param_sensitivity.csv"); meta = read_json("concept_meta.json")
    L, Lt = meta["L_traj"], meta["Ltilde_traj"]
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    for (s, k), g in d.groupby(["start_index", "direction"]):
        ax.plot(g.t, g.ratio, color=BLUE, alpha=0.3, lw=0.9)
    t = np.linspace(1e-3, 10, 300)
    ax.plot(t, Lt/L*(np.exp(L*t)-1), color=VERMILION, lw=1.8, label=r"$\frac{\tilde L}{L}(e^{Lt}-1)$ (Lemma 2 bound)")
    ax.set_yscale("log"); ax.set_xlabel("t"); ax.set_ylabel(r"$\|\varphi(t;p_1)-\varphi(t;p_2)\| / \|p_1-p_2\|$")
    ax.set_title("Lemma 2: flow sensitivity to the parameters"); ax.legend(loc="lower right"); ax.set_ylim(1e-3, 1e8)
    save(fig, "fig09_lemma2_param_sensitivity")

@figure
def fig10_prop1_node_removal():
    d = read_csv("prop1_node_removal.csv"); meta = read_json("concept_meta.json"); L = meta["L_traj"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9))
    cols = {"true": GREEN, "near": BLUE, "wrong": VERMILION}
    lab = {"true": r"$p=p^\star$", "near": r"$p$ near $p^\star$ ($\|p-p^\star\|=0.06$)", "wrong": r"$p$ wrong ($\|p-p^\star\|=0.04$, cubic term)"}
    ax = axs[0]
    for prm, g in d[d.experiment == "stride"].groupby("param"):
        ax.plot(g.n_removed, g.absdiff, marker="o", color=cols[prm], label=lab[prm])
    ax.set_xlabel(r"removed nodes $|I_R|$ (every $m$-th kept; $\Delta T_2=\Delta t$)")
    ax.set_ylabel(r"$|\hat J_K(p) - J_K(p)|$"); ax.set_yscale("log"); ax.set_title("Prop. 1: cost change vs nodes removed")
    ax.legend(fontsize=7); panel_label(ax, "a")
    ax = axs[1]
    for prm, g in d[d.experiment == "block"].groupby("param"):
        ax.plot(g.DeltaT2, g.absdiff, marker="s", color=cols[prm], label=lab[prm])
    x = np.linspace(1, 20, 50); ref = d[(d.experiment=="block")&(d.param=="near")].absdiff.iloc[0]
    ax.plot(x, ref*np.exp(L*(x-1))/1, color=LIGHTGREY, lw=1.2, ls="--", label=r"$\propto e^{L\Delta T_2}$: the bound's rate (loose)")
    ax.set_xlabel(r"$\Delta T_2=\tau_k-\tau_k^-$ (one block removed after node 30)")
    ax.set_ylabel(r"$|\hat J_K(p) - J_K(p)|$"); ax.set_yscale("log"); ax.set_ylim(1e-5, 1e2)
    ax.set_title(r"Prop. 1: cost change vs $\Delta T_2$"); ax.legend(fontsize=7); panel_label(ax, "b")
    save(fig, "fig10_prop1_node_removal")

# --------------------------------------------------------------- landscape --------
def _landscape2d(name, csv, xcol, ycol, xlabel, ylabel, truth):
    d = read_csv(csv); ks = sorted(d.window_size.unique())
    fig, axs = plt.subplots(1, len(ks), figsize=(7.2, 2.3), sharey=True)
    vmin = d.J[d.J < 1e3].min(); vmax = d.J[d.J < 1e3].quantile(0.98)
    for ax, k in zip(axs, ks):
        g = d[d.window_size == k]
        X = np.sort(g[xcol].unique()); Y = np.sort(g[ycol].unique())
        Z = g.pivot(index=ycol, columns=xcol, values="J").values
        Zm = np.ma.masked_where(Z >= 1e3, Z)
        im = ax.pcolormesh(X + truth[0], Y + truth[1], Zm, cmap=SEQ_CMAP + "_r", norm=LogNorm(vmin=vmin, vmax=vmax), shading="nearest")
        ax.contour(X + truth[0], Y + truth[1], Zm, levels=np.geomspace(vmin, vmax, 8), colors=GREY, linewidths=0.4)
        ax.pcolormesh(X + truth[0], Y + truth[1], np.ma.masked_where(Z < 1e3, np.ones_like(Z)), cmap="Greys", vmin=0, vmax=1.6, shading="nearest")
        ax.plot(truth[0], truth[1], marker="*", color=VERMILION, ms=9, mec="white", mew=0.6)
        ax.set_title(rf"$\kappa={k}$" + ("  (single shooting)" if k == 100 else ""), fontsize=9)
        ax.set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=axs, pad=0.01, shrink=0.9); cb.set_label(r"$J_\kappa(p)$ (grey = blow-up plateau)")
    save(fig, name, tight=False)

@figure
def fig11_landscape_2d_v_v3():
    _landscape2d("fig11_landscape_2d_v_v3", "landscape_2d_v_v3.csv", "dp_v", "dp_v3",
                 r"coefficient of $v$ in $\dot v$", r"coefficient of $v^3$ in $\dot v$", (1.0, -1/3))

@figure
def fig12_landscape_2d_w_wv():
    _landscape2d("fig12_landscape_2d_w_wv", "landscape_2d_w_wv.csv", "dp_w", "dp_wv",
                 r"coefficient of $w$ in $\dot v$", r"coefficient of $v$ in $\dot w$", (-1.0, 0.08))

@figure
def fig13_landscape_1d_slices():
    d = read_csv("landscape_1d_slices.csv"); m = read_csv("landscape_1d_minima.csv")
    ks = [1, 2, 5, 10, 25, 50, 100]
    fig = plt.figure(figsize=(7.2, 4.6)); gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[1, 1.15])
    cmap = plt.get_cmap(SEQ_CMAP)
    dirs = sorted(d.direction.unique())[:3]
    for j, dr in enumerate(dirs):
        ax = fig.add_subplot(gs[0, j])
        for i, k in enumerate(ks):
            g = d[(d.direction == dr) & (d.window_size == k)]
            J = g.J.values.copy(); J[J >= 1e3] = np.nan
            ax.plot(g.s, J, color=cmap(0.3 + 0.7*i/(len(ks)-1)), lw=1.1, label=rf"$\kappa={k}$")
        ax.set_yscale("log"); ax.set_xlabel(r"$s$"); ax.set_xlim(-1, 1)
        ax.set_title(rf"random direction $u_{dr}$", fontsize=9)
        if j == 0: ax.set_ylabel(r"$J_\kappa(p^\star + s\,u)$")
    ax = fig.add_subplot(gs[0, 3])
    mm = m.groupby("window_size").agg(mean=("n_local_minima", "mean"), lo=("n_local_minima", "min"), hi=("n_local_minima", "max")).reset_index()
    ax.plot(mm.window_size, mm["mean"], marker="o", color=BLUE); ax.fill_between(mm.window_size, mm.lo, mm.hi, color=BLUE, alpha=0.15)
    ax.set_xscale("log"); ax.set_xlabel(r"window size $\kappa$"); ax.set_ylabel("local minima per 1-D slice"); ax.set_title("mean (band: min–max) over 12 directions", fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    for i, k in enumerate(ks):
        g = d[(d.direction == dirs[0]) & (d.window_size == k)]
        J = g.J.values.copy(); J[J >= 1e3] = np.nan
        ax.plot(g.s, J, color=cmap(0.3 + 0.7*i/(len(ks)-1)), lw=1.3, label=rf"$\kappa={k}$")
    ax.axhspan(1e3, 3e3, color=LIGHTGREY, alpha=0.5); ax.text(-1.95, 1.3e3, "blow-up plateau (J = 1000)", fontsize=7, color=MUTED, va="center")
    ax.set_yscale("log"); ax.set_xlim(-2, 2); ax.set_ylim(5e-3, 3e3)
    ax.set_xlabel(r"$s$ along direction $u_{%d}$ (full range; blow-ups shown as gaps)" % dirs[0]); ax.set_ylabel(r"$J_\kappa$")
    ax.legend(ncol=7, fontsize=7, loc="upper center")
    save(fig, "fig13_landscape_1d_slices")

@figure
def fig14_hessian_conditioning():
    h = read_csv("hessian_at_ptrue.csv"); x = read_csv("landscape_x0_sensitivity.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axs[0]
    ax.plot(h.window_size, h.lambda_max, marker="o", color=BLUE, label=r"$\lambda_{\max}(H)$")
    ax.plot(h.window_size, np.abs(h.lambda_min), marker="s", color=VERMILION, label=r"$|\lambda_{\min}(H)|$")
    ax.plot(h.window_size, h.grad_norm, marker="^", color=GREEN, label=r"$\|\nabla J_\kappa(p^\star)\|$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_title(r"curvature of $J_\kappa$ at $p^\star$", fontsize=9)
    ax.legend(fontsize=7); panel_label(ax, "a")
    ax = axs[1]
    ax.bar(np.arange(len(h)), h.n_negative, color=VERMILION, width=0.6)
    ax.set_xticks(np.arange(len(h))); ax.set_xticklabels(h.window_size, fontsize=7); ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel("negative Hessian eigenvalues (of 20)"); ax.set_title(r"indefinite directions at $p^\star$", fontsize=9); panel_label(ax, "b")
    ax = axs[2]
    for k, c in zip((1, 10, 100), (BLUE, GREEN, VERMILION)):
        g = x[x.window_size == k]; J = g.J.values.copy(); J[J >= 1e3] = np.nan
        ax.plot(g.dv0, J, color=c, label=rf"$\kappa={k}$")
    ax.set_yscale("log"); ax.set_xlabel(r"perturbation $\delta$ of $v_0$"); ax.set_ylabel(r"$J_\kappa(x_0+\delta,\ p^\star)$")
    ax.set_title("sensitivity to the free initial condition", fontsize=9); ax.legend(fontsize=7); panel_label(ax, "c")
    save(fig, "fig14_hessian_conditioning")


# ------------------------------------------------------------------ sweeps --------
def _sweeps(exp=None):
    d = read_csv("sweeps.csv")
    return d if exp is None else d[d.exp == exp].copy()

def _arm_lines(ax, d, col, arms, ylog=True, seeds_thin=True, label=True):
    for arm in arms:
        g = d[d.arm == arm]
        if g.empty: continue
        if seeds_thin:
            for s, gs in g.groupby("seed"):
                gs = gs.sort_values("window_size"); ax.plot(gs.window_size, gs[col], color=ARM[arm], alpha=0.18, lw=0.8)
        m = g.groupby("window_size")[col].median()
        ax.plot(m.index, m.values, color=ARM[arm], marker=ARM_MARKER[arm], lw=1.8, label=ARM_LABEL[arm] if label else None, mec="white", mew=0.5)
    ax.set_xscale("log"); ax.set_xlabel(r"window size $\kappa$")
    if ylog: ax.set_yscale("log")

@figure
def fig15_sweep_headline():
    d = _sweeps("main"); arms = ["propagate", "best", "reset"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    _arm_lines(axs[0], d, "J", arms); axs[0].set_ylabel(r"$J_\kappa$ at the minimiser (incl. sparsity term)")
    axs[0].axhline(1e3, color=LIGHTGREY, lw=0.8); axs[0].text(1.05, 1.3e3, "blow-up plateau", fontsize=7, color=MUTED)
    axs[0].set_title("cost along the sweep", fontsize=9); axs[0].legend(fontsize=6.5, loc="center left"); panel_label(axs[0], "a")
    _arm_lines(axs[1], d, "p_err", arms, label=False); axs[1].set_ylabel(r"$\|p^{(\kappa)} - p^\star\|$")
    axs[1].set_title("parameter error (thin: 8 seeds; thick: median)", fontsize=8); panel_label(axs[1], "b")
    _arm_lines(axs[2], d, "score", arms, ylog=False, label=False); axs[2].set_ylabel("fractional recovery score (0–1)")
    axs[2].set_ylim(0, 1.02); axs[2].set_title("recovery score", fontsize=9); panel_label(axs[2], "c")
    save(fig, "fig15_sweep_headline")

@figure
def fig16_sweep_paired():
    d = _sweeps("main")
    piv = d.pivot_table(index=["seed", "window_size"], columns="arm", values="p_err").reset_index()
    rows = []
    for k, g in piv.groupby("window_size"):
        for other in ("reset", "best"):
            m, lo, hi = boot_ci(g["propagate"] - g[other])
            rows.append(dict(window_size=k, vs=other, mean=m, lo=lo, hi=hi, n=len(g), n_better=int((g["propagate"] < g[other]).sum()), n_tied=int((g["propagate"] == g[other]).sum())))
    r = pd.DataFrame(rows); r.to_csv(RES / "derived_paired_main.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axs[0]
    for other, c, off in (("reset", VERMILION, 0.95), ("best", GREEN, 1.05)):
        g = r[r.vs == other]
        ax.errorbar(g.window_size*off, g["mean"], yerr=[g["mean"]-g.lo, g.hi-g["mean"]], fmt=ARM_MARKER[other], color=c, capsize=2, lw=1.2,
                    label=f"GP − {ARM_LABEL[other]}")
    ax.axhline(0, color=GREY, lw=0.8); ax.set_xscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"paired $\Delta\|p-p^\star\|$ (mean, 95 % bootstrap CI)")
    ax.set_title("paired over 8 optimiser seeds", fontsize=9); ax.legend(fontsize=7); panel_label(ax, "a")
    ax = axs[1]
    g = r[r.vs == "reset"]
    ax.bar(np.arange(len(g)), g.n_better, color=BLUE, width=0.6, label="seeds where GP is strictly better")
    ax.bar(np.arange(len(g)), g.n_tied, bottom=g.n_better, color=LIGHTGREY, width=0.6, label="tied (identical result)")
    ax.set_xticks(np.arange(len(g))); ax.set_xticklabels(g.window_size, fontsize=6.5); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("number of seeds (of 8)")
    ax.set_title("cellwise: GP vs no propagation", fontsize=9); ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2); panel_label(ax, "b")
    save(fig, "fig16_sweep_paired")

def _coef_matrix(mins, exp, arm, seed=None):
    g = mins[(mins.exp == exp) & (mins.arm == arm)]
    if seed is not None: g = g[g.seed == seed]
    g = g[g["index"] > 2]                          # drop the 2 initial-condition entries
    piv = g.pivot_table(index="window_size", columns="index", values="value", aggfunc="median")
    return piv

@figure
def fig17_sweep_coefficients():
    mins = read_csv("sweep_minimizers.csv"); lib = read_csv("fhn_library.csv")
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[1, 4.5]))
    ax = axs[0]
    ax.bar(np.arange(20), lib.p_true, color=[BLUE]*10 + [ORANGE]*10, width=0.7); ax.axhline(0, color=GREY, lw=0.6)
    ax.set_ylabel(r"$p^\star$"); ax.set_title("true coefficients (blue: $\\dot v$ equation, orange: $\\dot w$ equation)", fontsize=9)
    ax = axs[1]
    P = _coef_matrix(mins, "main", "propagate")      # median over seeds
    E = P.values - lib.p_true.values[None, :]
    vmax = np.nanmax(np.abs(E[np.isfinite(E)])); vmax = min(vmax, 1.0)
    im = ax.imshow(E, cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(np.arange(len(P.index))); ax.set_yticklabels(P.index, fontsize=7); ax.set_ylabel(r"window size $\kappa$ (sweep order, top to bottom)")
    ax.set_xticks(np.arange(20)); ax.set_xticklabels(PARAM_LABELS, rotation=60, ha="right", fontsize=7)
    for j in range(20):
        for i in range(len(P.index)):
            if abs(E[i, j]) > 0.25: ax.text(j, i, f"{P.values[i,j]:.2f}", ha="center", va="center", fontsize=5.5, color=INK)
    cb = fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8); cb.set_label(r"$\mathrm{median}_{\rm seeds}\, p^{(\kappa)}_j - p^\star_j$")
    ax.set_title("guess propagation: seed-median coefficients along the sweep (cells off by > 0.25 annotated)", fontsize=9)
    save(fig, "fig17_sweep_coefficients")

@figure
def fig18_sweep_filmstrip():
    f = read_csv("sweep_fits.csv"); dat = read_csv("fhn_data.csv"); fine = read_csv("fhn_fine.csv"); sw = _sweeps("main")
    seed = 2 if (f.seed == 2).any() else 1
    ks = [1, 5, 25, 100]
    fig, axs = plt.subplots(2, len(ks), figsize=(7.2, 3.6), sharex=True, sharey="row")
    for j, k in enumerate(ks):
        for i, arm in enumerate(("propagate", "reset")):
            ax = axs[i, j]; g = f[(f.arm == arm) & (f.window_size == k) & (f.seed == seed)]
            ax.plot(fine.t, fine.v, color=GREY, lw=0.8, ls="--"); ax.plot(dat.t, dat.v, ".", color=GREY, ms=2.5)
            ax.plot(g.t, g.v, color=ARM[arm], lw=1.3)
            row = sw[(sw.arm == arm) & (sw.seed == seed) & (sw.window_size == k)]
            txt = rf"$\|p-p^\star\|$={row.p_err.iloc[0]:.2f}" + ("  blow-up" if row.blowup.iloc[0] else "") if len(row) else ""
            ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=6.5, va="top", color=ARM[arm])
            ax.set_ylim(-3, 3)
            if i == 0: ax.set_title(rf"$\kappa={k}$", fontsize=9)
            if i == 1: ax.set_xlabel("t")
        axs[0, 0].set_ylabel("v(t): guess propagation", color=ARM["propagate"]); axs[1, 0].set_ylabel("v(t): no propagation", color=ARM["reset"])
    fig.suptitle(f"single-shooting simulation from the minimiser found at each window size (optimiser seed {seed}); grey: truth and data", fontsize=8, y=1.0)
    save(fig, "fig18_sweep_filmstrip")

@figure
def fig19_sweep_traces():
    t = read_csv("traces_seed2.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)
    ks = sorted(t.window_size.unique()); cmap = plt.get_cmap(SEQ_CMAP)
    for ax, arm in zip(axs, ("propagate", "reset")):
        g = t[t.arm == arm]
        for i, k in enumerate(ks):
            gg = g[g.window_size == k]
            ax.plot(gg.iteration, gg.J, color=cmap(0.25 + 0.75*i/(len(ks)-1)), lw=0.9, label=rf"$\kappa$={k}" if k in (1, 5, 25, 100) else None)
        ax.set_yscale("log"); ax.set_xlabel("Nelder–Mead iteration"); ax.set_title(ARM_LABEL[arm], fontsize=9, color=ARM[arm])
    axs[0].set_ylabel(r"$J_\kappa$ (best simplex vertex)"); axs[0].legend(fontsize=7, title="light to dark: κ = 1 … 100", title_fontsize=7)
    save(fig, "fig19_sweep_traces")

@figure
def fig20_final_coefficients():
    mins = read_csv("sweep_minimizers.csv"); lib = read_csv("fhn_library.csv")
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    x = np.arange(20); w = 0.27
    ax.bar(x - w, lib.p_true, width=w, color=GREY, label=r"truth $p^\star$")
    for off, arm in ((0, "propagate"), (w, "reset")):
        g = mins[(mins.exp == "main") & (mins.arm == arm) & (mins.window_size == 100) & (mins["index"] > 2)]
        piv = g.pivot(index="seed", columns="index", values="value")
        ax.bar(x + off, piv.median(), width=w, color=ARM[arm], label=ARM_LABEL[arm] + " (seed median; dots: seeds)", alpha=0.9)
        for s in piv.index:
            ax.plot(x + off, piv.loc[s], ".", color=INK, ms=2.2, alpha=0.6)
    ax.axhline(0, color=GREY, lw=0.6); ax.set_xticks(x); ax.set_xticklabels(PARAM_LABELS, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("coefficient value"); ax.set_ylim(-1.7, 1.7); ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.set_title(r"coefficients at the end of the sweep ($\kappa=100$, single shooting) — 8 seeds", fontsize=9)
    save(fig, "fig20_final_coefficients")

def _var_panel(ax, d, groupcol, order, col, labelfmt, cmap_name=SEQ_CMAP, ylog=True, ref=None, refname=None):
    cmap = plt.get_cmap(cmap_name); n = len(order)
    for i, v in enumerate(order):
        g = d[d[groupcol] == v]
        if g.empty: continue
        m = g.groupby("window_size")[col].median()
        q1 = g.groupby("window_size")[col].quantile(0.25); q3 = g.groupby("window_size")[col].quantile(0.75)
        c = cmap(0.3 + 0.7*i/max(n-1, 1))
        ax.plot(m.index, m.values, marker="o", color=c, lw=1.5, label=labelfmt(v), mec="white", mew=0.4)
        ax.fill_between(m.index, q1.values, q3.values, color=c, alpha=0.12)
    if ref is not None:
        m = ref.groupby("window_size")[col].median(); ax.plot(m.index, m.values, color=VERMILION, ls="--", lw=1.4, label=refname)
    ax.set_xscale("log"); ax.set_xlabel(r"$\kappa$")
    if ylog: ax.set_yscale("log")

@figure
def fig21_schedule():
    d = _sweeps("schedule"); main = _sweeps("main"); main = main[main.arm == "propagate"].assign(arm="full")
    d = pd.concat([d, main]); order = ["dense", "full", "coarse", "jump"]
    names = {"dense": "dense: 44 steps (1,2,…,30,35,…,100)", "full": "full: 16 steps", "coarse": "coarse: 1, 5, 25, 100", "jump": "jump: 1 → 100"}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    _var_panel(axs[0], d, "arm", order, "p_err", lambda v: names[v]); axs[0].set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$ (median, IQR)"); axs[0].legend(fontsize=6.5); panel_label(axs[0], "a")
    _var_panel(axs[1], d, "arm", order, "score", lambda v: names[v], ylog=False); axs[1].set_ylabel("recovery score"); axs[1].set_ylim(0, 1.02); panel_label(axs[1], "b")
    ax = axs[2]
    fin = d[d.window_size == 100].groupby(["arm", "seed"]).agg(p_err=("p_err", "last"), wall=("wall_s", "first")).reset_index()
    tot = d.groupby(["arm", "seed"]).wall_s.sum().reset_index()
    for i, v in enumerate(order):
        g = fin[fin.arm == v]; tt = tot[tot.arm == v]
        ax.scatter(tt.wall_s, g.p_err, color=plt.get_cmap(SEQ_CMAP)(0.3 + 0.7*i/3), label=names[v].split(":")[0], s=18, edgecolor="white", lw=0.4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("total sweep wall time (s)"); ax.set_ylabel(r"final $\|p-p^\star\|$ at $\kappa=100$")
    ax.legend(fontsize=7); ax.set_title("cost vs benefit per seed", fontsize=9); panel_label(ax, "c")
    axs[0].set_title("window-size schedule (GP)", fontsize=9)
    save(fig, "fig21_schedule")

@figure
def fig22_penalty():
    d = pd.concat([_sweeps("penalty"), _sweeps("main")]); arms = ["propagate", "propagate_graded", "reset", "reset_graded"]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    _arm_lines(axs[0], d, "p_err", arms, seeds_thin=False); axs[0].set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$ (median)"); axs[0].legend(fontsize=6.5); panel_label(axs[0], "a")
    _arm_lines(axs[1], d, "score", arms, ylog=False, seeds_thin=False, label=False); axs[1].set_ylabel("recovery score"); axs[1].set_ylim(0, 1.02); panel_label(axs[1], "b")
    ax = axs[2]
    for arm in arms:
        g = d[d.arm == arm].groupby("window_size").blowup.mean()
        ax.plot(g.index, g.values, marker=ARM_MARKER[arm], color=ARM[arm], mec="white", mew=0.4)
    ax.set_xscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("fraction of seeds ending on the plateau"); ax.set_ylim(-0.02, 1.02); panel_label(ax, "c")
    axs[1].set_title("flat (repository) vs graded blow-up penalty", fontsize=9)
    save(fig, "fig22_penalty")

@figure
def fig23_noise():
    d = _sweeps("noise"); levels = sorted(d.noise.unique())
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, arm, lab in zip(axs[:2], ("propagate", "reset"), ("a", "b")):
        _var_panel(ax, d[d.arm == arm], "noise", levels, "p_err", lambda v: rf"$\sigma_{{\rm rel}}$={v:g}")
        ax.set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$ (median, IQR over 6 seeds)"); ax.set_title(ARM_LABEL[arm], fontsize=9, color=ARM[arm]); panel_label(ax, lab)
    axs[0].legend(fontsize=6.5)
    ax = axs[2]
    fin = d[d.window_size == 100]
    for arm in ("propagate", "reset"):
        g = fin[fin.arm == arm].groupby("noise").p_err
        ax.errorbar(levels, g.median(), yerr=[g.median()-g.quantile(0.25), g.quantile(0.75)-g.median()], marker=ARM_MARKER[arm], color=ARM[arm], capsize=2, label=ARM_LABEL[arm])
    ax.set_xscale("symlog", linthresh=0.01); ax.set_yscale("log"); ax.set_xlabel(r"relative noise level $\sigma_{\rm rel}$"); ax.set_ylabel(r"final $\|p-p^\star\|$ at $\kappa=100$")
    ax.legend(fontsize=7); panel_label(ax, "c"); ax.set_title("error vs noise", fontsize=9)
    save(fig, "fig23_noise")

@figure
def fig24_sparsity():
    d = _sweeps("sparsity"); mins = read_csv("sweep_minimizers.csv"); lib = read_csv("fhn_library.csv")
    gammas = sorted(d.gamma.unique())
    fig = plt.figure(figsize=(7.2, 4.4)); gs = gridspec.GridSpec(2, 3, figure=fig, height_ratios=[1, 1.3])
    ax = fig.add_subplot(gs[0, 0]); _var_panel(ax, d, "gamma", gammas, "p_err", lambda v: rf"$\gamma$={v:g}"); ax.set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$"); ax.legend(fontsize=6); panel_label(ax, "a")
    ax = fig.add_subplot(gs[0, 1]); _var_panel(ax, d, "gamma", gammas, "false_positives", lambda v: v, ylog=False); ax.set_ylabel("false non-zeros (of 14 true zeros)"); panel_label(ax, "b")
    ax = fig.add_subplot(gs[0, 2]); _var_panel(ax, d, "gamma", gammas, "score", lambda v: v, ylog=False); ax.set_ylabel("recovery score"); ax.set_ylim(0, 1.02); panel_label(ax, "c")
    ax = fig.add_subplot(gs[1, :])
    rows = []
    for gm in gammas:
        seeds = d[d.gamma == gm].seed.unique()
        g = mins[(mins.exp == "sparsity") & (mins.window_size == 100) & (mins["index"] > 2)]
        # sparsity jobs share arm 'propagate'; distinguish gamma via the sweeps table (same seed set) -> use derived join on arm+seed is ambiguous, so
        rows.append(None)
    # gamma is not stored in sweep_minimizers; rebuild via the per-gamma final minimisers written by the merge in sweeps.csv order is impossible,
    # so show the per-gamma median |p_j| from the runs whose (exp, seed) match and whose count equals len(gammas) blocks — see make_tables for the exact join.
    sub = mins[(mins.exp == "sparsity") & (mins.window_size == 100) & (mins["index"] > 2)].copy()
    sub["gamma"] = sub.arm.str.replace("gamma", "").astype(float)
    if len(sub):
        M = sub.groupby(["gamma", "index"]).value.median().unstack()
        im = ax.imshow(np.abs(M.values), cmap=SEQ_CMAP, vmin=0, vmax=1.0, aspect="auto")
        ax.set_yticks(np.arange(len(M.index))); ax.set_yticklabels([f"{v:g}" for v in M.index]); ax.set_ylabel(r"$\gamma$")
        ax.set_xticks(np.arange(20)); ax.set_xticklabels(PARAM_LABELS, rotation=60, ha="right", fontsize=7)
        for j, pt in enumerate(lib.p_true):
            if pt != 0: ax.add_patch(plt.Rectangle((j-0.5, -0.5), 1, len(M.index), fill=False, ec=VERMILION, lw=1.0))
        fig.colorbar(im, ax=ax, pad=0.01, shrink=0.8, label=r"median $|p_j|$ at $\kappa=100$ (red boxes: true support)")
    else:
        ax.text(0.5, 0.5, "per-γ minimisers not available in sweep_minimizers.csv", ha="center", transform=ax.transAxes)
    ax.set_title("sparsity weight γ: which coefficients survive", fontsize=9)
    save(fig, "fig24_sparsity")

@figure
def fig25_optimizers():
    d = pd.concat([_sweeps("optimizer"), _sweeps("main")]); d = d[d.arm.isin(["propagate", "reset", "bfgs_propagate", "lbfgs_propagate", "bfgs_reset", "lbfgs_reset"])]
    arms = ["propagate", "bfgs_propagate", "lbfgs_propagate", "reset", "bfgs_reset", "lbfgs_reset"]
    lab = {"propagate": "Nelder–Mead + GP", "reset": "Nelder–Mead, no propagation"}
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    for arm in arms:
        g = d[d.arm == arm]
        if g.empty: continue
        m = g.groupby("window_size").p_err.median()
        axs[0].plot(m.index, m.values, marker=ARM_MARKER[arm], color=ARM[arm], label=lab.get(arm, ARM_LABEL[arm]), mec="white", mew=0.4, ls="-" if "propagate" in arm else "--")
        w = g.groupby("window_size").wall_s.median()
        axs[1].plot(w.index, w.values, marker=ARM_MARKER[arm], color=ARM[arm], mec="white", mew=0.4, ls="-" if "propagate" in arm else "--")
        fc = g.groupby("window_size").f_calls.median()
        axs[2].plot(fc.index, fc.values, marker=ARM_MARKER[arm], color=ARM[arm], mec="white", mew=0.4, ls="-" if "propagate" in arm else "--")
    for ax in axs: ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\kappa$")
    axs[0].set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$ (median of 8 seeds)"); panel_label(axs[0], "a")
    axs[1].set_ylabel("wall time per window size (s)"); panel_label(axs[1], "b")
    axs[2].set_ylabel("objective evaluations per window size"); panel_label(axs[2], "c")
    fig.suptitle("optimiser: derivative-free vs AD-gradient (solid: GP, dashed: no propagation)", fontsize=8, y=1.02)
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.08))
    save(fig, "fig25_optimizers")

@figure
def fig26_basin():
    d = _sweeps("basin")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axs[0]
    for arm, c in (("k1", BLUE), ("k5", GREEN), ("k100", VERMILION)):
        g = d[d.arm == arm].groupby("radius")
        succ = g.p_err.apply(lambda x: (x < 0.3).mean())
        off = {"k1": 0.015, "k5": 0.0, "k100": -0.015}[arm]
        ax.plot(succ.index, succ.values + off, marker="o", color=c, label=rf"$\kappa$={arm[1:]}", mec="white", mew=0.4)
    ax.set_xscale("log"); ax.set_xlabel(r"start distance $r=\|p_0-p^\star\|$"); ax.set_ylabel(r"fraction of 10 starts with $\|p-p^\star\|<0.3$"); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7); ax.set_title("basin of attraction of one window size", fontsize=9); panel_label(ax, "a")
    ax = axs[1]
    for arm, c in (("k1", BLUE), ("k5", GREEN), ("k100", VERMILION)):
        g = d[d.arm == arm]
        ax.scatter(g.radius*(1 + 0.08*np.random.default_rng(1).standard_normal(len(g))), g.p_err, color=c, s=12, alpha=0.7, edgecolor="white", lw=0.3)
        ax.plot(g.groupby("radius").p_err.median().index, g.groupby("radius").p_err.median().values, color=c, lw=1.4)
    ax.plot([0.04, 1], [0.04, 1], color=LIGHTGREY, ls="--", lw=1); ax.text(0.5, 0.35, "no improvement", fontsize=7, color=MUTED, rotation=35)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"start distance $r$"); ax.set_ylabel(r"final $\|p-p^\star\|$ (dots: starts, line: median)")
    panel_label(ax, "b")
    save(fig, "fig26_basin")

@figure
def fig27_iters_substeps():
    it = _sweeps("iters"); ss = _sweeps("substeps")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    its = sorted(it.iters.unique())
    _var_panel(axs[0], it, "iters", its, "p_err", lambda v: f"{v} iterations"); axs[0].set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$ (median, IQR of 4 seeds)"); axs[0].legend(fontsize=6.5); axs[0].set_title("Nelder–Mead iteration budget", fontsize=9); panel_label(axs[0], "a")
    Ss = sorted(ss.S.unique())
    _var_panel(axs[1], ss, "S", Ss, "p_err", lambda v: rf"$S$={v} ($\delta t$={1/v:g})"); axs[1].set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$"); axs[1].legend(fontsize=6.5); axs[1].set_title("integrator sub-steps $S$", fontsize=9); panel_label(axs[1], "b")
    ax = axs[2]
    for i, S in enumerate(Ss):
        g = ss[ss.S == S].groupby("window_size").wall_s.median()
        ax.plot(g.index, g.values, marker="o", color=plt.get_cmap(SEQ_CMAP)(0.3 + 0.7*i/(len(Ss)-1)), label=rf"$S$={S}", mec="white", mew=0.4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("wall time per window size (s)"); ax.legend(fontsize=6.5); panel_label(ax, "c")
    save(fig, "fig27_iters_substeps")

@figure
def fig28_seedscale():
    d = _sweeps("seedscale")
    d["scale"] = d.arm.str.extract(r"sc([0-9.]+)_").astype(float); d["mode"] = d.arm.str.extract(r"_(propagate|reset)_"); d["pen"] = d.arm.str.extract(r"_(flat|graded)$")
    scales = sorted(d.scale.unique())
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, pen, lab in zip(axs[:2], ("flat", "graded"), ("a", "b")):
        g = d[(d.pen == pen) & (d["mode"] == "propagate")]
        _var_panel(ax, g, "scale", scales, "p_err", lambda v: rf"start scale {v:g}", ref=d[(d.pen == pen) & (d["mode"] == "reset") & (d.scale == 0.1)], refname="no propagation, scale 0.1")
        ax.set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$ (median, IQR of 8 seeds)"); ax.set_title(f"GP with {pen} penalty", fontsize=9); panel_label(ax, lab)
    axs[0].legend(fontsize=6)
    ax = axs[2]
    for pen, c, mk in (("flat", VERMILION, "s"), ("graded", PURPLE, "D")):
        g = d[(d.pen == pen) & (d["mode"] == "propagate") & (d.window_size == 1)].groupby("scale").blowup.mean()
        ax.plot(g.index, g.values, marker=mk, color=c, label=f"{pen} penalty", mec="white", mew=0.4)
    ax.set_xscale("log"); ax.set_xlabel(r"start scale (std of $p_0$ entries)"); ax.set_ylabel(r"fraction of seeds stuck on the plateau at $\kappa=1$"); ax.set_ylim(-0.02, 1.05)
    ax.legend(fontsize=7); ax.set_title("stuck on the plateau", fontsize=9); panel_label(ax, "c")
    save(fig, "fig28_seedscale")

# ------------------------------------------------------------------ solvers -------
@figure
def fig29_solvers_eval():
    d = read_csv("solvers_eval.csv"); d = d[d.param == d.param.unique()[0]].copy()
    d["method"] = d.method.str.replace("inhouse", "in-house").str.replace("diffeq_", "DiffEq ").str.replace("_", " ")
    d["setting"] = d.setting.astype(str).str.replace("abstol=reltol=", "tol ")
    d["label"] = d.method + " " + d.setting
    d = d[~d.method.str.contains("reference")]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    k5 = d[d.window_size == 5].copy(); k5["abs_err_vs_reference"] = k5.abs_err_vs_reference.clip(lower=1e-16)
    ax = axs[0]
    colors = [BLUE if "in-house" in m else (GREEN if "Tsit5" in m else VERMILION) for m in k5.method]
    ax.scatter(k5.time_ms, k5.abs_err_vs_reference, c=colors, s=22, edgecolor="white", lw=0.4)
    for _, r in k5.iterrows():
        ax.annotate(r.label.replace("in-house Tsit5 fixed", "in-house").replace("DiffEq ", "").replace(" adaptive", ""), (r.time_ms, r.abs_err_vs_reference), fontsize=5, xytext=(3, 2 if "S=5" not in r.label else -7), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"time per loss evaluation (ms), $\kappa=5$"); ax.set_ylabel(r"$|J - J_{\rm ref}|$ ($J_{\rm ref}$: adaptive Tsit5, tol $10^{-12}$)")
    ax.set_title(r"accuracy vs cost at $p^\star$ (blue: in-house fixed step)", fontsize=9); panel_label(ax, "a")
    ax = axs[1]
    for lbl, g in d.groupby("label"):
        if not (("in-house" in lbl and ("S=10" in lbl or "S=1 " in lbl or lbl.endswith("S=1") or "S=50" in lbl)) or "tol 1e-8" in lbl or "ImplicitEuler" in lbl): continue
        c = BLUE if "in-house" in lbl else (GREEN if "Tsit5" in lbl else VERMILION)
        ax.plot(g.window_size, g.time_ms, marker="o", ms=3, color=c, alpha=0.8 if "in-house" in lbl else 1, lw=1)
        ax.annotate(lbl.replace("in-house Tsit5 fixed", "").replace("DiffEq ", "").replace(" adaptive", ""), (100, g[g.window_size == 100].time_ms.iloc[0]), fontsize=5.5, xytext=(3, 0), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("time per loss evaluation (ms)"); ax.set_title("cost vs window size", fontsize=9); panel_label(ax, "b"); ax.set_xlim(0.8, 400)
    save(fig, "fig29_solvers_eval")

@figure
def fig30_solvers_sweep():
    d = read_csv("solvers_sweep.csv")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
    cols = {}
    for i, (m, g) in enumerate(d.groupby("method")):
        c = [BLUE, GREEN, VERMILION, PURPLE][i % 4]; cols[m] = c
        for s, gs in g.groupby("seed"):
            axs[0].plot(gs.window_size, gs.p_err, marker="o", ms=3, color=c, ls="-" if s == 2 else ":", label=f"{m} (seed {s})")
            axs[1].plot(gs.window_size, gs.wall_s, marker="o", ms=3, color=c, ls="-" if s == 2 else ":")
    for ax in axs: ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\kappa$")
    axs[0].set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$"); axs[0].legend(fontsize=5.5); axs[0].set_title("GP sweep with different integrators inside the loss (1000 NM iterations)", fontsize=8); panel_label(axs[0], "a")
    axs[1].set_ylabel("wall time per window size (s)"); panel_label(axs[1], "b")
    save(fig, "fig30_solvers_sweep")

# ------------------------------------------------------------------ bugs ----------
@figure
def fig31_bug_F2():
    d = read_csv("bug_F2_w_residual.csv"); s = read_csv("bug_F2_slice.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axs[0]
    for prm, g in d.groupby("param"):
        ax.plot(g.window_size, g.missing_fraction*100, marker="o", color=BLUE if prm == "ptrue" else LIGHTGREY, label=r"$p^\star$" if prm == "ptrue" else None, mec="white", mew=0.4)
    ax.plot([], [], color=LIGHTGREY, marker="o", label="3 perturbed p"); ax.set_xscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("share of the loss silently dropped (%)")
    ax.set_title("F2: share of the loss dropped", fontsize=8); ax.legend(fontsize=7); panel_label(ax, "a")
    ax = axs[1]
    g = d[d.param == "ptrue"]
    ax.plot(g.window_size, g.J_prefix_stiff, marker="s", color=VERMILION, label="as written (pre-fix)", mec="white", mew=0.4)
    ax.plot(g.window_size, g.J_fixed_stiff, marker="o", color=BLUE, label="fixed stiff branch", mec="white", mew=0.4)
    ax.plot(g.window_size, g.J_explicit, marker="^", color=GREEN, ls="--", label="explicit branch (reference)", mec="white", mew=0.4)
    ax.set_xscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$J_\kappa(p^\star)$"); ax.legend(fontsize=6.5); panel_label(ax, "b"); ax.set_title("the two branches agree once fixed", fontsize=8)
    ax = axs[2]
    ax.plot(s.p15, s.J_prefix_stiff, color=VERMILION, label="pre-fix: blind to the ẇ equation"); ax.plot(s.p15, s.J_fixed_stiff, color=BLUE, label="fixed")
    ax.axvline(0.08, color=GREY, lw=0.8, ls=":"); ax.set_yscale("log"); ax.set_xlabel(r"coefficient of $v$ in $\dot w$ (true 0.08), $\kappa=10$"); ax.set_ylabel(r"$J_{10}$")
    ax.legend(fontsize=6.5); panel_label(ax, "c"); ax.set_title("a 1-D slice through the ẇ equation", fontsize=8)
    save(fig, "fig31_bug_F2")

@figure
def fig32_bug_F1():
    d = read_csv("bug_F1_jacobian.csv"); meta = read_json("bug_F1_meta.json")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axs[0]
    ax.plot(d.t, d.J11_prefix, color=VERMILION, label=r"pre-fix (linear truncation)"); ax.plot(d.t, d.J11_fixed, color=BLUE, label="fixed"); ax.plot(d.t, d.J11_exact, color=GREEN, ls=":", lw=1.2, label="exact (AD of the true RHS)")
    ax.axhline(0, color=GREY, lw=0.6); ax.set_xlabel("t along the true trajectory"); ax.set_ylabel(r"$\partial \dot v/\partial v$"); ax.legend(fontsize=6.5); ax.set_title("F1: Jacobian entry used by the stability movies", fontsize=8); panel_label(ax, "a")
    ax = axs[1]
    ax.scatter(d.eig_re1_prefix, d.eig_im1_prefix, s=8, color=VERMILION, label="pre-fix eigenvalues"); ax.scatter(d.eig_re2_prefix, d.eig_im2_prefix, s=8, color=VERMILION)
    ax.scatter(d.eig_re1_fixed, d.eig_im1_fixed, s=8, color=BLUE, label="fixed eigenvalues"); ax.scatter(d.eig_re2_fixed, d.eig_im2_fixed, s=8, color=BLUE)
    ax.axvline(0, color=GREY, lw=0.6); ax.set_xlabel("Re λ"); ax.set_ylabel("Im λ"); ax.legend(fontsize=6.5); ax.set_title("eigenvalues along the trajectory", fontsize=8); panel_label(ax, "b")
    ax = axs[2]
    ax.plot(d.t, d.trace_prefix, color=VERMILION, label="pre-fix"); ax.plot(d.t, d.trace_fixed, color=BLUE, label="fixed"); ax.axhline(0, color=GREY, lw=0.6)
    ax.set_xlabel("t"); ax.set_ylabel(r"trace of the Jacobian (local divergence rate)"); ax.legend(fontsize=6.5); panel_label(ax, "c")
    ax.set_title(f"at u=(1.5,0.4): J11 = {meta.get('J11_prefix', float('nan')):.3g} vs {meta.get('J11_fixed', float('nan')):.3g}", fontsize=8)
    save(fig, "fig32_bug_F1")

@figure
def fig33_bug_F4():
    g = read_csv("bug_F4_gradient.csv"); t = read_csv("bug_F4_bfgs.csv")
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6))
    ax = axs[0]
    ax.plot(g.window_size, g.cosine_similarity, marker="o", color=BLUE, mec="white", mew=0.4)
    ax.set_xscale("log"); ax.set_xlabel(r"$\kappa$ of the objective optimised"); ax.set_ylabel(r"cosine of $\angle(g_\kappa, g_{100})$"); ax.set_ylim(0, 1.05)
    ax.set_title("F4: gradient received vs the right one", fontsize=8); panel_label(ax, "a")
    ax = axs[1]
    ax.plot(g.window_size, g.rel_diff, marker="s", color=VERMILION, mec="white", mew=0.4)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$\|g_\kappa - g_{100}\| / \|g_\kappa\|$"); panel_label(ax, "b"); ax.set_title("relative error of the wrong gradient", fontsize=8)
    ax = axs[2]
    for v, c in (("correct", BLUE), ("wrong", VERMILION)):
        gg = t[t.variant == v]; ax.plot(gg.iteration, gg.J_window, color=c, label=f"BFGS with {v} gradient")
    ax.set_yscale("log"); ax.set_xlabel("BFGS iteration"); ax.set_ylabel(r"$J_5$ ($\kappa=5$ objective)"); ax.legend(fontsize=6.5); panel_label(ax, "c")
    ax.set_title("what happens when the gradient is switched on", fontsize=8)
    save(fig, "fig33_bug_F4")

@figure
def fig34_bug_ordering_shadowing():
    o = read_csv("bug_ordering.csv"); s = read_csv("bug_shadowing.csv"); meta = read_json("bug_ordering_meta.json")
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ax = axs[0]; x = np.arange(20); w = 0.38
    ax.bar(x - w/2, o.p_true_new_order, width=w, color=BLUE, label="library order (data generated with this)")
    ax.bar(x + w/2, o.iloc[:, 4] if o.shape[1] > 4 else o.iloc[:, -1], width=w, color=VERMILION, label="what the old-order RHS reads at that index")
    ax.axhline(0, color=GREY, lw=0.6); ax.set_xticks(x); ax.set_xticklabels([f"{a}  |  {b}" for a, b in zip(o.new_label, o.old_label)], fontsize=5.5, rotation=90)
    ax.text(0.01, 0.02, "tick: library label | old-order label at the same index", transform=ax.transAxes, fontsize=6, color=MUTED)
    ax.set_ylabel("coefficient"); ax.legend(fontsize=6.5); ax.set_title(f"mixed monomial order in the mod 8 driver: spurious $\\|p-p^\\star\\|$ = {meta.get('spurious_p_err', float('nan')):.2f}", fontsize=8); panel_label(ax, "a")
    ax = axs[1]
    piv = s.pivot(index="dataset", columns="version", values="J")
    xx = np.arange(len(piv.index))
    ax.bar(xx - 0.2, piv["prefix"], width=0.4, color=VERMILION, label="pre-fix"); ax.bar(xx + 0.2, piv["fixed"], width=0.4, color=BLUE, label="fixed")
    ax.set_xticks(xx); ax.set_xticklabels(piv.index, fontsize=7); ax.set_ylabel(r"$J$ returned by forward_simulation_loss"); ax.set_yscale("log"); ax.legend(fontsize=6.5)
    ax.set_title("global-data shadowing: argument ignored pre-fix", fontsize=8); panel_label(ax, "b")
    save(fig, "fig34_bug_ordering_shadowing")

# ------------------------------------------------------------- other systems ------
def _other_panel(name, system, title, lab):
    d = read_csv("other_sweeps.csv"); d = d[d.system == system]; f = read_csv("other_fits.csv"); f = f[f.system == system]
    dat = read_csv(f"{system}_data.csv"); fine = read_csv(f"{system}_fine.csv"); lib = read_csv(f"{system}_library.csv")
    opts = sorted(d.optimizer.unique())
    ncol = 3 if len(opts) == 1 else 4
    fig, axs = plt.subplots(1, ncol, figsize=(7.2, 2.6))
    ax = axs[0]
    for opt in opts:
        for arm in ("propagate", "reset"):
            g = d[(d.optimizer == opt) & (d.arm == arm)]
            for s, gs in g.groupby("seed"): ax.plot(gs.window_size, gs.p_err, color=ARM[arm], alpha=0.2, lw=0.7, ls="-" if opt == opts[0] else "--")
            m = g.groupby("window_size").p_err.median()
            ax.plot(m.index, m.values, color=ARM[arm], marker=ARM_MARKER[arm], mec="white", mew=0.4, ls="-" if opt == opts[0] else "--", label=f"{ARM_LABEL[arm]}" + (f", {opt.upper()}" if len(opts) > 1 else ""))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$\|p^{(\kappa)}-p^\star\|$"); ax.legend(fontsize=5, loc="lower left"); ax.set_title(title, fontsize=8); panel_label(ax, "a")
    lo, hi = d.p_err.quantile(0.02), d.p_err.quantile(0.98); ax.set_ylim(lo*0.5, hi*3)
    ax = axs[1]
    for opt in opts:
        for arm in ("propagate", "reset"):
            m = d[(d.optimizer == opt) & (d.arm == arm)].groupby("window_size").score.median()
            ax.plot(m.index, m.values, color=ARM[arm], marker=ARM_MARKER[arm], mec="white", mew=0.4, ls="-" if opt == opts[0] else "--")
    ax.set_xscale("log"); ax.set_xlabel(r"$\kappa$"); ax.set_ylabel("recovery score (median)"); ax.set_ylim(0, 1.02); panel_label(ax, "b")
    opt0 = opts[-1] if system == "lorenz" else opts[0]
    ax = axs[2]
    ax.plot(fine.t, fine.iloc[:, 1], color=GREY, ls="--", lw=0.8); ax.plot(dat.t, dat.iloc[:, 1], ".", color=GREY, ms=2.5)
    for arm in ("propagate", "reset"):
        g = f[(f.arm == arm) & (f.optimizer == opt0) & (f.window_size == 100)]
        ax.plot(g.t, g.x1, color=ARM[arm], lw=1.2, label=ARM_LABEL[arm])
    ax.set_xlabel("t"); ax.set_ylabel(lab + rf" fit, $\kappa$=100, seed 1" + (f" ({opt0.upper()})" if len(opts) > 1 else "")); ax.legend(fontsize=5.5, loc="lower right"); panel_label(ax, "c")
    ymax = np.nanmax(np.abs(dat.iloc[:, 1]))*1.6; ax.set_ylim(-ymax if dat.iloc[:, 1].min() < 0 else -0.1*ymax, ymax)
    if ncol == 4:
        ax = axs[3]; mins = read_csv("other_minimizers.csv"); d_ = 2 if system == "lv" else 3
        for i, opt in enumerate(opts):
            g = mins[(mins.system == system) & (mins.arm == "propagate") & (mins.optimizer == opt) & (mins.window_size == 100) & (mins["index"] > d_)]
            piv = g.pivot(index="seed", columns="index", values="value").median()
            ax.plot(np.arange(len(piv)), piv.values, marker=".", ms=3, lw=0.8, color=[GREEN, BLUE][i % 2], label=f"GP, {opt.upper()}")
        ax.plot(np.arange(len(lib)), lib.p_true, "k_", ms=6, label="truth"); ax.axhline(0, color=GREY, lw=0.5)
        ax.set_xlabel("coefficient index"); ax.set_ylabel("value (seed median)"); ax.legend(fontsize=5.5); panel_label(ax, "d")
    save(fig, name)

@figure
def fig35_lotka_volterra():
    _other_panel("fig35_lotka_volterra", "lv", "Lotka–Volterra (12 coefficients, 5 % noise)", "x(t)")

@figure
def fig36_lorenz():
    _other_panel("fig36_lorenz", "lorenz", "Lorenz (30 coefficients, 2 % noise)", "x(t)")

# ------------------------------------------------------------------ post ----------
@figure
def fig37_post_hessian_path():
    h = read_csv("post_hessian_at_minimizers.csv"); p = read_csv("post_path_between_minimizers.csv"); x = read_csv("post_cross_evaluation.csv")
    h = h[h.seed != 1]; p = p[p.seed == 2]
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ax = axs[0]
    for s, g in h.groupby("seed"):
        ax.scatter(g.window_size, g.lambda_min, color=np.where(g.n_negative == 0, BLUE, VERMILION), s=10, alpha=0.7, edgecolor="white", lw=0.3)
    frac_pd = (h.n_negative == 0).mean()
    ax.axhline(0, color=GREY, lw=0.6); ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"$\lambda_{\min}$ of the data-term Hessian at $p^{(\kappa)}$")
    ax.set_title(f"positive definite in {frac_pd*100:.0f} % of cells", fontsize=8); panel_label(ax, "a")
    ax.scatter([], [], color=BLUE, s=10, label="all eigenvalues > 0"); ax.scatter([], [], color=VERMILION, s=10, label="some < 0"); ax.legend(fontsize=6, loc="lower left")
    ax = axs[1]
    cmap = plt.get_cmap(SEQ_CMAP); pairs = list(p.groupby(["kappa_from", "kappa_to"]))
    for i, ((a, b), g) in enumerate(pairs):
        ax.plot(g.s, g.J_next/g.J_next.iloc[0], color=cmap(0.25 + 0.75*i/max(len(pairs)-1, 1)), lw=0.9)
    ax.set_yscale("log"); ax.set_xlabel(r"$s$: from $p^{(\kappa_i)}$ ($s=0$) to $p^{(\kappa_{i+1})}$ ($s=1$)"); ax.set_ylabel(r"$J_{\kappa_{i+1}}$ along the segment, relative to $s=0$")
    ax.set_title("carried guess to next minimiser (seed 2)", fontsize=8); panel_label(ax, "b")
    ax = axs[2]
    piv = x.pivot(index="kappa_min", columns="kappa_eval", values="J"); Z = np.log10(piv.values)
    im = ax.imshow(Z, cmap=SEQ_CMAP + "_r", aspect="auto"); ax.set_xticks(np.arange(len(piv.columns))); ax.set_xticklabels(piv.columns, fontsize=5.5, rotation=90)
    ax.set_yticks(np.arange(len(piv.index))); ax.set_yticklabels(piv.index, fontsize=5.5); ax.set_xlabel(r"evaluated with window size $\kappa_j$"); ax.set_ylabel(r"minimiser of $\kappa_i$")
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8, label=r"$\log_{10} J_{\kappa_j}(p^{(\kappa_i)})$"); ax.set_title(f"cross-evaluation (seed {int(x.seed.iloc[0])})", fontsize=8); panel_label(ax, "c")
    save(fig, "fig37_post_hessian_path")

if __name__ == "__main__":
    names = sys.argv[1:] or list(FIGS)
    for n in names:
        try:
            FIGS[n]()
        except (FileNotFoundError, KeyError, IndexError, ValueError) as e:
            import traceback; traceback.print_exc()
            print(f"SKIP {n}: {type(e).__name__}: {e}")
