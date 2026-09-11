"""concept_figures.py — R002 concept figures (fig01–fig07).

Slide-/report-quality CONCEPT figures that introduce the inverse problem,
single vs multiple shooting, the shooting partition, node removal and guess
propagation.  Every number plotted comes from `analysis/results/`; the only
synthetic curves are in fig07, which is labelled "schematic" on its face.

Run:  /opt/anaconda3/bin/python3 figures/concept_figures.py
Writes figures/figNN_*.pdf + .png.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import *                                              # noqa: F401,F403

import numpy as np
import pandas as pd
from matplotlib.patches import ConnectionPatch, Rectangle
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# shared constants / colours
# ----------------------------------------------------------------------------
WIN_A = "#E7F0F8"        # very light blue   (odd windows)
WIN_B = "#F1F1F1"        # very light grey   (even windows)
DATA_C = INK             # noisy data: dark dots
TRUTH_C = GREY           # true trajectory: grey dashed
WRONG_C = VERMILION      # model integrated at the wrong p
NODE_C = BLACK

DATA = read_csv("fhn_data.csv")
FINE = read_csv("fhn_fine.csv")
VF = read_csv("fhn_vector_field.csv")
LIB = read_csv("fhn_library.csv")
SEG = read_csv("concept_segments.csv")
COST = read_csv("concept_costs.csv")
SLICE = read_csv("landscape_1d_slices.csv")
MINIMA = read_csv("landscape_1d_minima.csv")
CMETA = read_json("concept_meta.json")
DMETA = read_json("fhn_data_meta.json")

L_TRAJ = CMETA["L_traj"]            # 3.0515  Lipschitz constant along the orbit
LT_TRAJ = CMETA["Ltilde_traj"]      # 10.5645 parameter-Lipschitz constant
ETA = DMETA["eta_max_norm"]         # 0.15868 max per-sample noise 2-norm
DT = DMETA["Delta_t"]               # 1.0
NOISE_J = DMETA["noise_floor_J"]    # 0.00515
BLOWUP = 1000.0                     # ms_loss blow-up plateau value


def cost(param, ws):
    return float(COST[(COST.param == param) & (COST.window_size == ws)].J.iloc[0])


def segments(param, ws):
    """Yield (window_index, DataFrame) for each shooting window, in time order."""
    s = SEG[(SEG.param == param) & (SEG.window_size == ws)]
    for k in sorted(s.window.unique()):
        yield int(k), s[s.window == k].sort_values("t")


def seg_residuals(param, ws):
    """Per-datum residual ||y_i - x_m(t_i)|| for every datum strictly inside a
    window, i.e. t_i in (tau_{k-1}, tau_k] — the nodes themselves have residual 0
    by construction and are excluded."""
    out = []
    for k, g in segments(param, ws):
        t0, t1 = g.t.min(), g.t.max()
        d = DATA[(DATA.t > t0 + 1e-9) & (DATA.t < t1 + 1e-9)]
        if len(d) == 0:
            continue
        r = np.hypot(d.v.values - np.interp(d.t, g.t, g.v),
                     d.w.values - np.interp(d.t, g.t, g.w))
        out.append(pd.DataFrame({"t": d.t.values, "r": r, "window": k}))
    return pd.concat(out, ignore_index=True).sort_values("t")


def interp_seg(g, t, col):
    return np.interp(t, g.t, g[col])


def bracket(ax, x0, x1, y, text, up=True, tick=None, color=INK, fs=8,
            lw=0.9, pad=None, ha="center"):
    """Square bracket spanning [x0, x1] at height y with a centred label."""
    sgn = 1.0 if up else -1.0
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    tick = 0.022 * span if tick is None else tick
    pad = 0.030 * span if pad is None else pad
    ax.plot([x0, x0, x1, x1], [y - sgn * tick, y, y, y - sgn * tick],
            color=color, lw=lw, clip_on=False, solid_capstyle="butt", zorder=6)
    ax.text(0.5 * (x0 + x1), y + sgn * pad, text, ha=ha,
            va="bottom" if up else "top", fontsize=fs, color=color, zorder=6)


def shade_windows(ax, param, ws, alpha=1.0, zorder=0):
    for k, g in segments(param, ws):
        ax.axvspan(g.t.min(), g.t.max(), color=WIN_A if k % 2 else WIN_B,
                   lw=0, alpha=alpha, zorder=zorder)


# ============================================================================
# fig01 — the inverse problem
# ============================================================================
def fig01():
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.12, 0.52],
                          width_ratios=[1.05, 1.0], hspace=0.60, wspace=0.22,
                          left=0.078, right=0.985, top=0.915, bottom=0.055)

    # ---- (a) time series --------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.plot(FINE.t, FINE.v, ls="--", lw=1.0, color=TRUTH_C, zorder=1,
            label="true trajectory")
    ax.plot(FINE.t, FINE.w, ls="--", lw=1.0, color=TRUTH_C, zorder=1)
    ax.plot(DATA.t, DATA.v, "o", ms=2.9, color=STATE["v"], mec="none", zorder=3,
            label="$v$ samples $y_i$")
    ax.plot(DATA.t, DATA.w, "o", ms=2.9, color=STATE["w"], mec="none", zorder=3,
            label="$w$ samples $y_i$")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2.5, 3.10)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("state")
    ax.legend(loc="lower left", ncol=3, fontsize=8, handlelength=1.7,
              columnspacing=1.3, borderpad=0.2, bbox_to_anchor=(-0.006, 1.01))
    ax.text(1.0, 1.01, "$N+1 = 101$ samples,  $\\Delta t = 1$,  5 % relative noise",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom", ha="right")
    # Delta-t bracket on one data interval (t = 30 -> 31)
    ti, tj = 30.0, 31.0
    yb = 2.32
    ax.plot([ti, tj], [yb, yb], color=INK, lw=1.0, zorder=6)
    for x in (ti, tj):
        ax.plot([x, x], [yb - 0.13, yb + 0.13], color=INK, lw=1.0, zorder=6)
    ax.plot([ti, tj], np.interp([ti, tj], DATA.t, DATA.v), "o", ms=6.0, mfc="none",
            mec=INK, mew=1.0, zorder=6)
    ax.plot([ti, ti], [yb - 0.13, float(np.interp(ti, DATA.t, DATA.v)) + 0.16],
            color=INK, lw=0.5, ls=":", zorder=5)
    ax.plot([tj, tj], [yb - 0.13, float(np.interp(tj, DATA.t, DATA.v)) + 0.16],
            color=INK, lw=0.5, ls=":", zorder=5)
    ax.text(0.5 * (ti + tj), yb + 0.20, r"one data interval $\Delta t = 1$",
            fontsize=8, color=INK, ha="center", va="bottom")
    panel_label(ax, "a", x=-0.058, y=1.13)

    # ---- (b) phase plane --------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    gv, gw = VF.v.values, VF.w.values
    du, dw = VF.dv.values, VF.dw.values
    nrm = np.hypot(du, dw)
    nrm[nrm == 0] = 1.0
    ax.quiver(gv, gw, du / nrm, dw / nrm, color=LIGHTGREY, angles="xy",
              scale=34, width=0.0032, headwidth=3.6, headlength=4.0, zorder=0)
    ax.plot(FINE.v, FINE.w, ls="--", lw=1.1, color=TRUTH_C, zorder=2, label="true orbit")
    ax.plot(DATA.v, DATA.w, "o", ms=3.0, color=DATA_C, mec="none", zorder=3,
            label="noisy samples")
    ax.set_xlabel("$v$")
    ax.set_ylabel("$w$")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-1.22, 1.80)
    ax.set_yticks([0.0, 0.5, 1.0, 1.5])
    ax.legend(loc="lower center", fontsize=8, handlelength=1.6, borderpad=0.25,
              labelspacing=0.2, frameon=True, facecolor="white", edgecolor="none",
              framealpha=0.92)
    ax.text(-2.5, 1.76, "grey arrows: vector field of the true FHN model",
            fontsize=8, color=MUTED, va="top", ha="left")
    panel_label(ax, "b", x=-0.205, y=1.04)

    # ---- model / library statement ----------------------------------------
    axt = fig.add_subplot(gs[1, 1])
    axt.axis("off")
    axt.text(0.0, 1.0,
             "Data-generating model (FitzHugh–Nagumo)\n"
             "   $\\dot v = v - v^3/3 - w + 0.5$\n"
             "   $\\dot w = (v + 0.7 - 0.8\\,w)\\,/\\,12.5$\n"
             "\n"
             "Ansatz  $\\dot x_i = \\sum_{j} p_{ij}\\,\\Theta_j(v,w)$,  where\n"
             "$\\Theta$ = every monomial in $(v,w)$ of degree $\\leq 3$\n"
             "(10 terms per equation $\\Rightarrow$ 20 unknowns $p_{ij}$).",
             transform=axt.transAxes, fontsize=8.5, va="top", ha="left", color=INK,
             linespacing=1.7)

    # ---- (c) sparse coefficient matrix ------------------------------------
    ax = fig.add_subplot(gs[2, :])
    P = np.zeros((2, 10))
    for _, r in LIB.iterrows():
        i = int(r["index"]) - 1
        P[i // 10, i % 10] = r["p_true"]
    signed_sqrt = np.sign(P) * np.sqrt(np.abs(P) / np.abs(P).max())
    M = np.ma.masked_where(P == 0.0, signed_sqrt)
    cmap = plt.get_cmap(DIV_CMAP).copy()
    cmap.set_bad("white")
    ax.imshow(M, cmap=cmap, norm=TwoSlopeNorm(vmin=-1.05, vcenter=0.0, vmax=1.05),
              aspect="auto", interpolation="nearest")
    for i in range(2):
        for j in range(10):
            if P[i, j] != 0.0:
                txt = f"{P[i, j]:.3f}".rstrip("0").rstrip(".")
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if abs(signed_sqrt[i, j]) > 0.62 else INK,
                        fontweight="bold")
    for j in range(11):
        ax.axvline(j - 0.5, color="white", lw=1.4)
    for i in range(3):
        ax.axhline(i - 0.5, color="white", lw=1.4)
    ax.set_xticks(range(10))
    ax.set_xticklabels(MONO_LABELS, fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([r"$\dot v$", r"$\dot w$"], fontsize=10)
    ax.set_xlabel("library monomial $\\Theta_j(v,w)$")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("unknown sparse coefficients $p$  (20 entries, 7 non-zero)",
                 fontsize=9.5, pad=6)
    ax.text(1.0, -0.70, "blank cell = true zero;  cell colour = sign of $p_{ij}$, "
                        "shaded by $\\sqrt{|p_{ij}|}$",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="top", ha="right")
    panel_label(ax, "c", x=-0.058, y=1.06)

    fig.suptitle("The inverse problem: recover the few non-zero entries of $p$ "
                 "from noisy samples of $v,w$", fontsize=10, y=0.995)
    save(fig, "fig01_problem", tight=False)


# ============================================================================
# fig02 — single shooting at the wrong p
# ============================================================================
def fig02():
    fig = plt.figure(figsize=(7.2, 4.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.85, 1.0], hspace=0.22,
                          top=0.865, bottom=0.155, left=0.095, right=0.985)
    ax = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    _, g = next(iter(segments("wrong", 100)))
    vm = interp_seg(g, DATA.t.values, "v")

    ax.plot(FINE.t, FINE.v, ls="--", lw=1.0, color=TRUTH_C, zorder=1,
            label="true trajectory $v(t)$")
    ax.vlines(DATA.t, DATA.v, vm, color=MUTED, lw=0.55, alpha=0.85, zorder=2)
    ax.plot(g.t, g.v, "-", lw=1.6, color=WRONG_C, zorder=3,
            label=r"single shooting from $y_0$ at wrong $p$")
    ax.plot(DATA.t, DATA.v, "o", ms=3.0, color=DATA_C, mec="none", zorder=4,
            label="noisy data $y_i$")
    ax.plot([0], [DATA.v.iloc[0]], "o", ms=7.5, mfc="none", mec=WRONG_C, mew=1.5,
            zorder=5, clip_on=False)
    ax.annotate("$y_0$", (0, DATA.v.iloc[0]), xytext=(-1.5, -12),
                textcoords="offset points", fontsize=9, color=WRONG_C, ha="center")
    ax.set_ylabel("$v$")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-2.85, 3.35)
    ax.tick_params(labelbottom=False)
    ax.legend(loc="upper left", ncol=3, fontsize=8, handlelength=1.7,
              columnspacing=1.2, borderpad=0.2, bbox_to_anchor=(-0.005, 1.20))
    # divergence annotation
    ax.annotate("", xy=(27.5, 2.45), xytext=(15.5, -0.25),
                arrowprops=dict(arrowstyle="-|>", color=WRONG_C, lw=1.3,
                                connectionstyle="arc3,rad=-0.28"))
    ax.text(28.8, 2.55, "residuals grow like $e^{Lt}$\nuntil the model slips phase",
            fontsize=8.5, color=WRONG_C, va="center", ha="left")
    ax.text(0.985, 0.965,
            "$\\mathcal{J}_{\\kappa=100}(p_{\\mathrm{wrong}}) = "
            f"{cost('wrong', 100):.2f}$\n"
            "$\\mathcal{J}_{\\kappa=100}(p^{*}) = "
            f"{cost('true', 100):.4f}$",
            transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec=LIGHTGREY, lw=0.8))
    panel_label(ax, "a", x=-0.075, y=1.20)

    # ---- residual panel ---------------------------------------------------
    r = np.hypot(DATA.v.values - vm, DATA.w.values - interp_seg(g, DATA.t.values, "w"))
    m = DATA.t.values > 0                       # t = 0 is the node: residual is 0
    ax2.semilogy(DATA.t.values[m], r[m], "o-", ms=2.8, lw=1.0, color=WRONG_C,
                 label=r"$\|y_i - x_m(t_i)\|$")
    ax2.axhline(ETA, ls=":", lw=1.1, color=INK)
    ax2.text(100.0, ETA * 1.25, f"noise level $\\|\\eta\\|_{{\\max}} = {ETA:.3f}$",
             fontsize=8, color=INK, va="bottom", ha="right")
    # observed exponential growth over the divergence window
    sel = (DATA.t.values >= 16) & (DATA.t.values <= 22)
    lam, b = np.polyfit(DATA.t.values[sel], np.log(r[sel]), 1)
    tt = np.linspace(14, 24, 50)
    ax2.semilogy(tt, np.exp(b + lam * tt), ls="--", lw=1.2, color=GREY)
    ax2.annotate(f"observed growth $\\propto e^{{\\lambda t}}$, $\\lambda = {lam:.2f}$\n"
                 f"(the Lipschitz bound $L = {L_TRAJ:.2f}$ is an upper bound)",
                 (20.5, float(np.exp(b + lam * 20.5))), xytext=(3.0, 55.0),
                 textcoords="data", fontsize=8, color=GREY, va="center", ha="left",
                 arrowprops=dict(arrowstyle="->", color=GREY, lw=0.8, shrinkB=3))
    ax2.set_xlabel("time $t$")
    ax2.set_ylabel(r"$\|y_i - x_m(t_i)\|$")
    ax2.set_ylim(5e-2, 6e2)
    ax2.set_yticks([1e-1, 1e0, 1e1, 1e2])
    panel_label(ax2, "b", x=-0.075, y=1.02)

    fig.suptitle("Single shooting: one initial condition, error compounding over the "
                 "whole record", fontsize=10, y=0.985)
    fig.text(0.5, 0.012,
             "$p_{\\mathrm{wrong}}$: the true $p^{*}$ with the $v^3$ coefficient "
             f"$-1/3 \\rightarrow {CMETA['p_wrong_v3']}$ and the $\\dot w$ block "
             f"scaled by {CMETA['p_wrong_w_scale']}"
             f"   ($\\|p-p^{{*}}\\| = {CMETA['p_wrong_err']:.4f}$);  "
             f"noise floor $\\mathcal{{J}} = {NOISE_J:.4f}$",
             fontsize=8, color=MUTED, va="bottom", ha="center")
    save(fig, "fig02_single_shooting", tight=False)


# ============================================================================
# fig03 — the shooting partition and node removal (schematic, no data)
# ============================================================================
def fig03():
    N, KAPPA = 24, 4
    t = np.arange(N + 1, dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.3))
    fig.subplots_adjust(hspace=1.05, top=0.82, bottom=0.07, left=0.030, right=0.985)

    def timeline(ax, kappa, node_sym, removed=None):
        nodes = t[::kappa]
        K = len(nodes) - 1
        ax.set_xlim(-2.2, N + 10.5)
        ax.set_ylim(-1.05, 1.45)
        ax.axis("off")
        for k in range(K):                              # alternating window shading
            ax.add_patch(Rectangle((nodes[k], -0.30), nodes[k + 1] - nodes[k], 0.60,
                                   fc=WIN_A if k % 2 == 0 else WIN_B, ec="none", zorder=0))
        ax.annotate("", xy=(N + 2.0, 0), xytext=(-1.6, 0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2), zorder=3)
        ax.vlines(t, -0.10, 0.10, color=INK, lw=0.9, zorder=3)          # data ticks
        for i, lab in [(0, "$t_0$"), (1, "$t_1$"), (2, "$t_2$"), (N, "$t_N$")]:
            ax.text(t[i], -0.21, lab, ha="center", va="top", fontsize=8, color=INK)
        ax.text(0.5 * N, -0.56, r"$N$ data intervals of width $\Delta t$",
                ha="center", va="top", fontsize=8, color=MUTED)
        # shooting nodes
        ax.vlines(nodes, -0.34, 0.50, color=node_sym["c"], lw=1.7, zorder=4)
        ax.plot(nodes, np.full_like(nodes, 0.50), "o", ms=6.5,
                color=node_sym["c"], mec=node_sym["c"], mfc=node_sym["c"], zorder=5)
        for k, x in enumerate(nodes):
            ax.text(x, 0.60, node_sym["lab"].format(k), ha="center", va="bottom",
                    fontsize=8.5, color=node_sym["c"])
        if removed is not None:
            ax.vlines(removed, -0.34, 0.50, color=MUTED, lw=1.1, ls=(0, (2, 2)), zorder=4)
            ax.plot(removed, np.full_like(removed, 0.50), "o", ms=6.5, mfc="white",
                    mec=MUTED, mew=1.2, zorder=5)
            for x in removed:
                ax.plot([x], [0.50], "x", ms=4.5, color=MUTED, mew=1.2, zorder=6)
            ax.text(removed[0], 0.62, "removed", ha="center", va="bottom",
                    fontsize=8, color=MUTED)
        return nodes, K

    # ---- row 1: fine partition -------------------------------------------
    ax = axes[0]
    nodes, K = timeline(ax, KAPPA, dict(c=BLUE, lab=r"$\tau_{{{}}}$"))
    bracket(ax, nodes[0], nodes[1], 1.12, r"window length $\Delta T = \kappa\,\Delta t$",
            up=True, tick=0.11, pad=0.05)
    bracket(ax, t[0], t[1], -0.72, r"$\Delta t$", up=False, tick=0.09, pad=0.04)
    ax.text(N + 2.8, 0.02,
            f"$K = {K}$ windows,\n$\\kappa = {KAPPA}$ data\nintervals each",
            fontsize=8.5, color=BLUE, va="center", ha="left", linespacing=1.4)
    ax.set_title(r"Fine partition: a shooting node $\tau_k$ at every $\kappa$-th datum; "
                 r"each window restarts from the datum $y_{\tau_{k-1}}$",
                 fontsize=9.5, loc="left", pad=24, color=INK)
    panel_label(ax, "a", x=-0.015, y=1.34)

    # ---- row 2: coarse partition -----------------------------------------
    ax = axes[1]
    nodes2, K2 = timeline(ax, 2 * KAPPA, dict(c=VERMILION, lab=r"$\tilde\tau_{{{}}}$"),
                          removed=t[KAPPA::2 * KAPPA])
    bracket(ax, nodes2[0], nodes2[1], 1.12,
            r"new window length $= 2\kappa\,\Delta t = \Delta T_1 + \Delta T_2$", up=True,
            tick=0.11, pad=0.05)
    ax.text(N + 2.8, 0.02,
            f"$K/2 = {K2}$ windows,\n$2\\kappa = {2 * KAPPA}$ data\nintervals each",
            fontsize=8.5, color=VERMILION, va="center", ha="left", linespacing=1.4)
    ax.set_title(r"Node removal: delete every other node $\Rightarrow$ half as many windows, "
                 r"each twice as long",
                 fontsize=9.5, loc="left", pad=24, color=INK)
    panel_label(ax, "b", x=-0.015, y=1.34)

    save(fig, "fig03_partition", tight=False)


# ============================================================================
# fig04 — single vs multiple shooting at the same wrong p
# ============================================================================
def fig04():
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.30,
                          top=0.895, bottom=0.075, left=0.095, right=0.985)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharex=axA)
    axC = fig.add_subplot(gs[2], sharex=axA)

    def traj_panel(ax, ws, title, letter, show_windows=True):
        if show_windows:
            shade_windows(ax, "wrong", ws)
        ax.plot(FINE.t, FINE.v, ls="--", lw=0.9, color=TRUTH_C, zorder=1)
        for k, g in segments("wrong", ws):
            vm = interp_seg(g, DATA.t.values, "v")
            m = (DATA.t.values >= g.t.min() - 1e-9) & (DATA.t.values <= g.t.max() + 1e-9)
            ax.vlines(DATA.t.values[m], DATA.v.values[m], vm[m], color=MUTED,
                      lw=0.5, alpha=0.85, zorder=2)
            ax.plot(g.t, g.v, "-", lw=1.5, color=WRONG_C, zorder=3)
            ax.plot([g.t.iloc[0]], [g.v.iloc[0]], "o", ms=5.0, mfc="white",
                    mec=NODE_C, mew=1.3, zorder=5)
        ax.plot(DATA.t, DATA.v, "o", ms=2.6, color=DATA_C, mec="none", zorder=4)
        ax.set_ylabel("$v$")
        ax.set_xlim(-3, 103)
        ax.set_ylim(-2.9, 3.0)
        ax.set_title(title, fontsize=9.5, loc="left", pad=5, color=INK)
        ax.text(0.985, 0.955,
                f"$\\mathcal{{J}}_{{\\kappa={ws}}}(p_{{\\mathrm{{wrong}}}}) = "
                f"{cost('wrong', ws):.3f}$",
                transform=ax.transAxes, fontsize=8.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=LIGHTGREY, lw=0.8))
        panel_label(ax, letter, x=-0.075, y=1.04)

    traj_panel(axA, 100, r"Single shooting  ($\kappa = 100$, one window, $K = 1$): "
                         r"error accumulates over $t_N = 100$", "a", show_windows=False)
    axA.plot([], [], ls="--", lw=0.9, color=TRUTH_C, label="true trajectory")
    axA.plot([], [], "-", lw=1.5, color=WRONG_C, label=r"model at wrong $p$")
    axA.plot([], [], "o", ms=2.6, color=DATA_C, label="noisy data")
    axA.plot([], [], "o", ms=5.0, mfc="white", mec=NODE_C, mew=1.3,
             label=r"shooting node $\tau_k$ (restart at $y_{\tau_k}$)")
    axA.legend(loc="lower left", ncol=4, fontsize=8, handlelength=1.6,
               columnspacing=1.1, borderpad=0.2, bbox_to_anchor=(-0.005, 1.13))
    axA.tick_params(labelbottom=False)

    traj_panel(axB, 10, r"Multiple shooting  ($\kappa = 10$, $K = 10$ windows): "
                        r"every window restarts from its own datum", "b")
    axB.tick_params(labelbottom=False)

    # ---- (c) residual magnitude per datum ---------------------------------
    for ws, col, mk in ((100, VERMILION, "o"), (10, BLUE, "s"), (5, GREEN, "^")):
        rr = seg_residuals("wrong", ws)
        axC.semilogy(rr.t, rr.r, mk + "-", ms=2.6, lw=0.9, color=col, alpha=0.95,
                     label=f"$\\kappa = {ws}$  ($\\mathcal{{J}} = {cost('wrong', ws):.3f}$)")
    axC.axhline(ETA, ls=":", lw=1.1, color=INK)
    axC.text(1.0, ETA * 1.25, f"$\\|\\eta\\|_{{\\max}} = {ETA:.3f}$", fontsize=8,
             color=INK, va="bottom", ha="left")
    axC.set_xlabel("time $t$")
    axC.set_ylabel(r"$\|y_i - x_m(t_i)\|$")
    axC.set_ylim(3e-3, 4e3)
    axC.set_yticks([1e-2, 1e-1, 1e0, 1e1, 1e2])
    axC.legend(loc="lower left", ncol=3, fontsize=8, handlelength=1.7,
               columnspacing=1.0, borderpad=0.2, bbox_to_anchor=(-0.005, 1.01))
    panel_label(axC, "c", x=-0.075, y=1.12)
    axC.text(0.985, 0.94,
             "small $\\kappa$ keeps the cost near the noise floor: smoother\n"
             "landscape, but less able to tell $p_{\\mathrm{wrong}}$ from $p^{*}$",
             transform=axC.transAxes, fontsize=8, color=MUTED, va="top", ha="right",
             linespacing=1.4)

    fig.suptitle(r"Restarting at the nodes caps the error growth at $e^{L\Delta T}$ "
                 r"instead of $e^{L t_N}$  (same wrong $p$ in every panel)",
                 fontsize=10, y=0.985)
    save(fig, "fig04_multiple_shooting", tight=False)


# ============================================================================
# fig05 — removing one node
# ============================================================================
def fig05():
    T0, T1 = 19.0, 41.0
    TKM, TK, TKP = 20.0, 25.0, 30.0          # tau_k^-, tau_k (removed), tau_k^+
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(top=0.90, bottom=0.44, left=0.075, right=0.985)

    # alternating shading of the COARSE (kappa = 10) windows in view
    for k, g in segments("wrong", 10):
        a, b = g.t.min(), g.t.max()
        if b <= T0 or a >= T1:
            continue
        ax.axvspan(max(a, T0), min(b, T1), color=WIN_A if k % 2 else WIN_B, lw=0, zorder=0)

    ax.plot(FINE.t, FINE.v, ls="--", lw=1.0, color=TRUTH_C, zorder=1,
            label="true trajectory")
    for k, g in segments("wrong", 5):                    # fine: kappa = 5
        if g.t.max() <= T0 or g.t.min() >= T1:
            continue
        ax.plot(g.t, g.v, "-", lw=1.7, color=BLUE, zorder=3)
        ax.plot([g.t.iloc[0]], [g.v.iloc[0]], "o", ms=6.0, color=BLUE, mec="white",
                mew=0.9, zorder=6)
    for k, g in segments("wrong", 10):                   # coarse: kappa = 10
        if g.t.max() <= T0 or g.t.min() >= T1:
            continue
        lw = 2.6 if abs(g.t.min() - TKM) < 1e-9 else 1.7
        ax.plot(g.t, g.v, "-", lw=lw, color=VERMILION, zorder=4)
        ax.plot([g.t.iloc[0]], [g.v.iloc[0]], "s", ms=6.0, color=VERMILION,
                mec="white", mew=0.9, zorder=6)
    d = DATA[(DATA.t >= T0) & (DATA.t <= T1)]
    ax.plot(d.t, d.v, "o", ms=3.6, color=DATA_C, mec="none", zorder=5,
            label="noisy data $y_i$")
    ax.plot([], [], "-", lw=1.7, color=BLUE, label=r"fine partition, $\kappa = 5$")
    ax.plot([], [], "-", lw=1.7, color=VERMILION, label=r"coarse partition, $\kappa = 10$")

    # the removed node
    vk = float(np.interp(TK, DATA.t, DATA.v))
    ax.plot([TK], [vk], "o", ms=10.0, mfc="white", mec=MUTED, mew=1.6, zorder=7)
    ax.plot([TK], [vk], "x", ms=6.0, color=MUTED, mew=1.6, zorder=8)
    ax.axvline(TK, color=MUTED, ls=(0, (2, 2)), lw=1.0, zorder=2)
    for x in (TKM, TKP):
        ax.axvline(x, color=VERMILION, lw=0.9, alpha=0.55, zorder=2)

    ax.set_xlim(T0, T1)
    ax.set_ylim(-3.15, 4.35)
    ax.set_xlabel("time $t$")
    ax.set_ylabel("$v$")
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_xticks([20, 25, 30, 35, 40])
    ax.set_xticklabels(["20\n$\\tau_k^-$", "25\n$\\tau_k$", "30\n$\\tau_k^+$",
                        "35", "40"])
    for lbl, c in zip(ax.get_xticklabels(), (VERMILION, MUTED, VERMILION, GREY, GREY)):
        lbl.set_color(c)
        lbl.set_fontsize(9)
    ax.set_xlabel("time $t$", labelpad=2)
    ax.legend(loc="upper left", ncol=1, fontsize=8, handlelength=1.7,
              borderpad=0.25, labelspacing=0.28, bbox_to_anchor=(0.005, 0.99),
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)

    ax.annotate(r"removed node $\tau_k$", (TK, vk), xytext=(25.9, 0.35),
                textcoords="data", fontsize=8.5, color=MUTED, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9, shrinkB=9,
                                connectionstyle="arc3,rad=0.25"))
    # manuscript convention: Delta T_1 = max_k(tau_k - tau_{k-1})   (fine window length)
    #                        Delta T_2 = max_k(tau_k - tau_k^-)     (removed node -> left
    #                                                                neighbour in the new
    #                                                                partition)
    bracket(ax, TKM, TK, -1.55, r"$\Delta T_2$", up=True, tick=0.16, pad=0.05)
    bracket(ax, TK, TKP, -1.55, r"$\Delta T_1$", up=True, tick=0.16, pad=0.05)
    bracket(ax, TKM, TKP, -2.45, "", up=True, tick=0.16, pad=0.05)
    ax.text(0.5 * (TKM + TKP), -2.40,
            r"coarse window $= \Delta T_1 + \Delta T_2 = 10$", ha="center",
            va="bottom", fontsize=8, color=INK, zorder=7,
            bbox=dict(boxstyle="square,pad=0.16", fc="white", ec="none"))
    gc = SEG[(SEG.param == "wrong") & (SEG.window_size == 10) &
             (SEG.window == 4)].sort_values("t")          # coarse window [30, 40]
    ax.annotate("the coarse segment drifts further from\nthe data before it is reset "
                "(gap 0.57 at $t = 37$)",
                (37.0, float(np.interp(37.0, gc.t, gc.v))),
                xytext=(34.6, 2.75), textcoords="data", fontsize=8.5, color=VERMILION,
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="->", color=VERMILION, lw=0.9,
                                connectionstyle="arc3,rad=-0.25", shrinkB=4))

    fig.text(0.5, 0.985,
             r"Removing the node $\tau_k$: the segment launched from $\tau_k^-$ must now "
             r"cover $[\tau_k^-,\tau_k^+]$",
             fontsize=10, ha="center", va="top", color=INK)
    fig.text(0.5, 0.245,
             r"$\|\hat{\mathcal{J}}_K(p)-\mathcal{J}_K(p)\| \leq "
             r"2\,|\mathcal{I}_R|\,C_{\max}\left("
             r"e^{L(\Delta T_1+\Delta T_2)}\|\eta\|_{\max}"
             r"+e^{L\Delta T_1}\left[\frac{\tilde{L}}{L}\left(e^{L\Delta T_2}-1\right)"
             r"\|p-p^{*}\|+\|\eta\|_{\max}\right]\right)$",
             fontsize=9.5, ha="center", va="top", color=INK,
             bbox=dict(boxstyle="round,pad=0.45", fc="#FBFBFB", ec=LIGHTGREY, lw=0.9))
    fig.text(0.5, 0.115,
             r"$\Delta T_1=\max_k(\tau_k-\tau_{k-1})$ = fine window length;  "
             r"$\Delta T_2=\max_k(\tau_k-\tau_k^-)$ = distance from the removed node "
             r"to its left neighbour" "\n"
             r"(both $=5$ here);  $|\mathcal{I}_R|$ = number of removed nodes;  "
             r"$C_{\max}$ = largest segment/data magnitude over the affected windows"
             "\n"
             r"$L = %.2f$, $\tilde L = %.2f$ (Lipschitz in $x$, in $p$);  "
             r"$\|\eta\|_{\max} = %.3f$.   "
             r"Background shading: the coarse ($\kappa=10$) windows."
             % (L_TRAJ, LT_TRAJ, ETA),
             fontsize=8, ha="center", va="top", color=MUTED, linespacing=1.55)
    save(fig, "fig05_node_removal", tight=False)


# ============================================================================
# fig06 — guess propagation across increasingly rugged landscapes
# ============================================================================
def fig06():
    # direction with the most local minima at kappa = 100
    m100 = MINIMA[MINIMA.window_size == 100]
    DIR = int(m100.loc[m100.n_local_minima.idxmax(), "direction"])
    KS = [1, 5, 25, 100]

    fig = plt.figure(figsize=(7.2, 4.3))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 0.42], hspace=0.58, wspace=0.16,
                          left=0.085, right=0.985, top=0.855, bottom=0.045)
    axes = [fig.add_subplot(gs[0, j]) for j in range(4)]
    YLO, YHI = 8e-3, 6e3

    mins = []
    for j, (ax, k) in enumerate(zip(axes, KS)):
        g = SLICE[(SLICE.direction == DIR) & (SLICE.window_size == k)].sort_values("s")
        s, J = g.s.values, g.J.values
        fin = J < BLOWUP
        ax.axhspan(3e2, YHI, facecolor="none", edgecolor=LIGHTGREY, hatch="////",
                   lw=0.0, zorder=0)
        ax.axhline(BLOWUP, color=MUTED, lw=0.9, ls="-")
        Jp = np.where(fin, J, np.nan)
        ax.semilogy(s, Jp, "-", lw=1.2, color=BLUE, zorder=3)
        ax.semilogy(s, np.where(fin, np.nan, BLOWUP), "-", lw=2.0, color=MUTED, zorder=3)
        # local minima (recomputed here; counts match landscape_1d_minima.csv)
        loc = [i for i in range(1, len(J) - 1)
               if fin[i] and fin[i - 1] and fin[i + 1] and J[i] < J[i - 1] and J[i] <= J[i + 1]]
        ax.semilogy(s[loc], J[loc], "o", ms=4.2, mfc="none", mec=VERMILION, mew=1.0,
                    zorder=4)
        i0 = int(np.nanargmin(Jp))
        ax.semilogy([s[i0]], [J[i0]], "*", ms=11, color=GREEN, mec="white", mew=0.6,
                    zorder=6)
        mins.append((s[i0], J[i0]))
        nref = int(MINIMA[(MINIMA.direction == DIR) &
                          (MINIMA.window_size == k)].n_local_minima.iloc[0])
        ax.set_title(f"$\\kappa = {k}$", fontsize=9.5, pad=14, color=INK)
        ax.text(0.5, 1.01, f"{nref} local min.", transform=ax.transAxes, fontsize=8,
                color=VERMILION, ha="center", va="bottom")
        ax.set_xlim(-2.05, 2.05)
        ax.set_ylim(YLO, YHI)
        ax.set_xticks([-2, -1, 0, 1, 2])
        ax.set_xlabel("$s$")
        if j == 0:
            ax.set_ylabel(r"$\mathcal{J}_\kappa(p^{*}+s\,u)$")
        else:
            ax.tick_params(labelleft=False)
        panel_label(ax, "abcd"[j], x=-0.10 if j else -0.34, y=1.15)

    axes[0].text(-1.95, 2.2e3, "blow-up plateau  $\\mathcal{J}=10^3$", fontsize=8,
                 color=MUTED, va="center", ha="left",
                 bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
    axes[0].semilogy([], [], "o", ms=4.2, mfc="none", mec=VERMILION, mew=1.0,
                     label="local minimum")
    axes[0].semilogy([], [], "*", ms=9, color=GREEN, label="global min. of slice")
    axes[0].legend(loc="upper right", fontsize=8, handlelength=1.0, borderpad=0.25,
                   labelspacing=0.25, handletextpad=0.3, bbox_to_anchor=(1.03, 0.735),
                   frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)

    # ---- "initial guess" arrows between panels ----------------------------
    for j in range(3):
        cp = ConnectionPatch(xyA=mins[j], coordsA=axes[j].transData,
                             xyB=mins[j + 1], coordsB=axes[j + 1].transData,
                             arrowstyle="-|>", mutation_scale=11, lw=1.3, color=GREEN,
                             connectionstyle="arc3,rad=-0.42", shrinkA=7, shrinkB=7,
                             zorder=20)
        cp.set_clip_on(False)
        fig.add_artist(cp)
    fig.text(0.535, 0.312, r"$p_{\mathrm{guess}}$ carried forward: each minimiser "
                           r"becomes the next initial guess",
             fontsize=8.5, color=GREEN, ha="center", va="top")

    # ---- pseudo-code box --------------------------------------------------
    axp = fig.add_subplot(gs[1, :])
    axp.axis("off")
    code = ("$p_{\\mathrm{guess}} \\leftarrow$ random\n"
            "for $\\kappa = 1, 2, \\ldots, N$   (a few nodes removed per step)\n"
            "     $p^{(\\kappa)} \\leftarrow \\arg\\min\\ \\mathcal{J}_\\kappa(p)$   "
            "starting from $p_{\\mathrm{guess}}$\n"
            "     $p_{\\mathrm{guess}} \\leftarrow p^{(\\kappa)}$\n"
            "end;   polish with single shooting ($\\kappa = N$) from $p_{\\mathrm{guess}}$")
    axp.text(0.005, 1.02, code, transform=axp.transAxes, fontsize=8.4, va="top",
             ha="left", color=INK, linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.45", fc="#FBFBFB", ec=LIGHTGREY, lw=0.9))
    axp.text(0.995, 1.02,
             "Small windows $\\Rightarrow$ smooth, nearly\n"
             "convex landscape (1 minimum at $\\kappa=1$).\n"
             "Large windows $\\Rightarrow$ rugged "
             f"({int(m100.n_local_minima.max())} minima at $\\kappa=100$).\n"
             "The minimiser is carried into each\n"
             "rougher landscape instead of restarting.",
             transform=axp.transAxes, fontsize=8.2, va="top", ha="right", color=MUTED,
             linespacing=1.45)

    fig.text(0.5, 0.985,
             f"Guess propagation: 1-D cost slices along random unit direction "
             f"$u$ (direction {DIR} of 12), $p^{{*}}$ at $s=0$",
             fontsize=10, ha="center", va="top", color=INK)
    save(fig, "fig06_guess_propagation", tight=False)


# ============================================================================
# fig07 — error-growth schematic
# ============================================================================
def fig07():
    L, eta, TEND, DTW = L_TRAJ, ETA, 10.0, 2.0
    t = np.linspace(0, TEND, 2001)
    single = eta * np.exp(L * t)
    tau = np.floor(t / DTW) * DTW
    multi = eta * np.exp(L * np.minimum(t - tau, DTW))
    cap = eta * np.exp(L * DTW)

    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    fig.subplots_adjust(top=0.80, bottom=0.145, left=0.105, right=0.735)

    for k in range(int(TEND / DTW)):
        ax.axvspan(k * DTW, (k + 1) * DTW, color=WIN_A if k % 2 == 0 else WIN_B,
                   lw=0, zorder=0)
    ax.semilogy(t, single, "-", lw=1.9, color=VERMILION, zorder=3)
    ax.semilogy(t, multi, "-", lw=1.9, color=BLUE, zorder=4)
    ax.axhline(cap, ls="--", lw=1.0, color=BLUE, zorder=2)
    ax.axhline(eta, ls=":", lw=1.1, color=INK, zorder=2)
    ax.plot(np.arange(0, TEND, DTW), np.full(int(TEND / DTW), eta), "o", ms=5.5,
            mfc="white", mec=NODE_C, mew=1.2, zorder=6, clip_on=False)

    ax.set_xlim(0, TEND)
    ax.set_ylim(eta * 0.12, 3e13)
    ax.set_xticks(np.arange(0, TEND + 1, DTW))
    ax.set_xlabel("time $t$")
    ax.set_ylabel(r"bound on $\|x_m(t)-x(t)\|$")
    ax.set_yticks([1e-1, 1e2, 1e5, 1e8, 1e11])

    label_end(ax, TEND, single[-1], "single shooting:\n"
              r"$e^{Lt}\,\|\eta\|_{\max}$", VERMILION, fontweight="bold")
    label_end(ax, TEND, cap, "multiple shooting,\n"
              r"restart every $\Delta T$:" + "\n"
              r"capped at $e^{L\Delta T}\|\eta\|_{\max}$" + f" $= {cap:.0f}$",
              BLUE, fontweight="bold")
    label_end(ax, TEND, eta,
              r"restart error $\|\eta\|_{\max}$" + f" $= {eta:.3f}$", INK)
    ax.annotate(r"restart at each node $\tau_k$", (2 * DTW, eta), xytext=(4.55, 0.032),
                textcoords="data", fontsize=8.5, color=NODE_C, ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=NODE_C, lw=0.8, shrinkB=5,
                                connectionstyle="arc3,rad=0.2"))
    ax.text(0.012, 0.965, "SCHEMATIC — analytic bounds, not measured residuals",
            transform=ax.transAxes, fontsize=8.5, color=VERMILION, va="top", ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=VERMILION, lw=0.8))
    ax.text(0.012, 0.875,
            f"$L = {L:.2f}$ (Lipschitz along the orbit),  "
            f"$\\|\\eta\\|_{{\\max}} = {eta:.3f}$,  $\\Delta T = {DTW:.0f}$\n"
            "The bounds are loose: the real residual saturates at the\n"
            "attractor scale (see fig02b, fig04c).",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="top", ha="left",
            linespacing=1.45)

    fig.text(0.5, 0.985,
             r"Restarting every $\Delta T$ replaces one $e^{Lt_N}$ excursion by a "
             r"sawtooth capped at $e^{L\Delta T}$",
             fontsize=10, ha="center", va="top", color=INK)
    save(fig, "fig07_error_growth_schematic", tight=False)


if __name__ == "__main__":
    fig01(); fig02(); fig03(); fig04(); fig05(); fig06(); fig07()
