"""make_animations.py — MP4 animations for FHN and Lotka–Volterra (report addendum).

Frames are rendered with matplotlib into figures/anim_frames/<name>/ and encoded with the
ffmpeg binary shipped as a Julia artifact (FFMPEG_jll; no system ffmpeg in the sandbox).
Outputs (figures/):
  anim_fhn_sweep.mp4              GP vs control along the sweep: v(t) fits + cost/error (seed 2)
  anim_fhn_phase.mp4              phase-plane fit + coefficient bars along the sweep (seed 2)
  anim_fhn_multiple_shooting.mp4  concept: single shooting drawn in time, then windows, then node removal
  anim_fhn_landscape.mp4          2-D cost landscape morphing with κ, with the GP / control paths
  anim_lv_sweep.mp4               Lotka–Volterra: x(t), y(t) fits along the sweep (seed 1)
  anim_lv_phase.mp4               Lotka–Volterra phase plane + coefficient bars (seed 1)
  anim_lv_landscape.mp4           Lotka–Volterra cost landscape (x, xy plane of ẋ) morphing with κ, GP / control paths
  anim_lv_landscape_x2_xy.mp4     same in the (x², xy) plane of ẋ, 161×161 (true-zero quadratic term: finite-time blow-up)
  anim_lv_landscape_xy_xy.mp4     same in the (xy in ẋ, xy in ẏ) interaction plane, 161×161
Usage: python make_animations.py [name ...]
"""
import sys, pathlib, subprocess, os, glob, shutil
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import *
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LogNorm

FFMPEG = "/Users/manu_jay/.julia/artifacts/ba7aa2eabe225711e391274d206e636166078211/bin/ffmpeg"
JL = "/Applications/Julia-1.11.app/Contents/Resources/julia/lib"
def ffenv():
    libs = [JL, JL + "/julia"] + glob.glob(os.path.expanduser("~/.julia/artifacts/*/lib"))
    e = dict(os.environ); e["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(libs); return e
def encode(frames_dir, out, in_fps, out_fps=24):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-framerate", str(in_fps), "-i", str(frames_dir / "frame_%04d.png"),
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p", "-r", str(out_fps), "-c:v", "libx264", "-crf", "20", "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True, env=ffenv()); print("wrote", out.name)
def frames_dir(name):
    d = OUT / "anim_frames" / name; shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); return d
def savef(fig, d, i):
    fig.savefig(d / f"frame_{i:04d}.png", dpi=110); plt.close(fig)
ANIMS = {}
def anim(fn): ANIMS[fn.__name__] = fn; return fn

FULL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100]
LV_MONO = ["1", "y", "y²", "x", "xy", "x²"]; LV_LABELS = [f"ẋ: {m}" for m in LV_MONO] + [f"ẏ: {m}" for m in LV_MONO]
LV_P = [0, 0, 0, 1, -0.5, 0, 0, -0.8, 0, 0, 0.3, 0]

def _fhn_common():
    dat = read_csv("fhn_data.csv"); fine = read_csv("fhn_fine.csv"); sw = read_csv("sweeps.csv"); sw = sw[(sw.exp == "main") & (sw.seed == 2)]
    mins = read_csv("sweep_minimizers.csv"); mins = mins[(mins.exp == "main") & (mins.seed == 2)]
    return dat, fine, sw, mins

@anim
def anim_fhn_sweep():
    f = read_csv("sweep_fits.csv"); f = f[f.seed == 2]; dat, fine, sw, _ = _fhn_common(); d = frames_dir("fhn_sweep")
    for i, k in enumerate(FULL):
        fig = plt.figure(figsize=(10, 4.6)); gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.7, 1])
        for r, arm in enumerate(("propagate", "reset")):
            ax = fig.add_subplot(gs[r, 0])
            ax.plot(fine.t, fine.v, color=GREY, lw=0.9, ls="--", label="truth"); ax.plot(dat.t, dat.v, ".", color=GREY, ms=3, label="data")
            g = f[(f.arm == arm) & (f.window_size == k)]; ax.plot(g.t, g.v, color=ARM[arm], lw=1.8, label=ARM_LABEL[arm])
            nodes = dat.t.values[::k]; ax.plot(nodes, np.full_like(nodes, -2.8), "|", color=ARM[arm], ms=7, mew=1.3)
            ax.set_ylim(-3, 3); ax.set_ylabel("v(t)", fontsize=11); ax.legend(fontsize=9, loc="upper right", ncol=3)
            row = sw[(sw.arm == arm) & (sw.window_size == k)]
            if len(row): ax.text(0.01, 0.04, rf"$J_\kappa$ = {row.J.iloc[0]:.3g}    $\|p-p^\star\|$ = {row.p_err.iloc[0]:.2f}" + ("   (blow-up plateau)" if row.blowup.iloc[0] else ""), transform=ax.transAxes, fontsize=10, color=ARM[arm])
            if r == 0: ax.set_title(rf"window size $\kappa$ = {k}   ({int(np.ceil(100/k))} windows; ticks = shooting nodes)", fontsize=12)
            else: ax.set_xlabel("t", fontsize=11)
        for r, col in enumerate(("J", "p_err")):
            ax = fig.add_subplot(gs[r, 1])
            for arm in ("propagate", "reset"):
                g = sw[sw.arm == arm].sort_values("window_size"); gg = g[g.window_size <= k]
                ax.plot(g.window_size, g[col], color=ARM[arm], alpha=0.2, lw=1); ax.plot(gg.window_size, gg[col], color=ARM[arm], marker="o", ms=4, lw=1.8)
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel(r"cost $J_\kappa$" if col == "J" else r"$\|p^{(\kappa)}-p^\star\|$", fontsize=11); ax.axvline(k, color=LIGHTGREY, lw=0.8)
            if r == 1: ax.set_xlabel(r"$\kappa$", fontsize=11)
        fig.tight_layout(); savef(fig, d, i)
    encode(d, OUT / "anim_fhn_sweep.mp4", 1/1.4)

@anim
def anim_fhn_phase():
    f = read_csv("sweep_fits.csv"); f = f[f.seed == 2]; dat, fine, sw, mins = _fhn_common(); lib = read_csv("fhn_library.csv"); d = frames_dir("fhn_phase")
    for i, k in enumerate(FULL):
        fig = plt.figure(figsize=(10, 4.6)); gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1.5])
        ax = fig.add_subplot(gs[:, 0])
        ax.plot(fine.v, fine.w, color=GREY, lw=0.9, ls="--", label="true orbit"); ax.plot(dat.v, dat.w, ".", color=GREY, ms=3, label="data")
        for arm in ("propagate", "reset"):
            g = f[(f.arm == arm) & (f.window_size == k)]; ax.plot(g.v, g.w, color=ARM[arm], lw=1.6, label=ARM_LABEL[arm])
        ax.set_xlim(-2.7, 2.7); ax.set_ylim(-0.8, 2.0); ax.set_xlabel("v", fontsize=11); ax.set_ylabel("w", fontsize=11); ax.legend(fontsize=9, loc="upper left")
        ax.set_title(rf"phase plane, single-shooting simulation from the minimiser at $\kappa$ = {k}", fontsize=11)
        for r, arm in enumerate(("propagate", "reset")):
            ax = fig.add_subplot(gs[r, 1]); x = np.arange(20)
            ax.bar(x - 0.2, lib.p_true, width=0.4, color=GREY, label=r"truth $p^\star$")
            g = mins[(mins.arm == arm) & (mins.window_size == k) & (mins["index"] > 2)].sort_values("index")
            ax.bar(x + 0.2, g.value.values, width=0.4, color=ARM[arm], label=ARM_LABEL[arm])
            row = sw[(sw.arm == arm) & (sw.window_size == k)]
            ax.text(0.99, 0.95, rf"$\|p-p^\star\|$ = {row.p_err.iloc[0]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=10, color=ARM[arm])
            ax.axhline(0, color=GREY, lw=0.6); ax.set_ylim(-1.6, 1.6); ax.set_xticks(x); ax.set_xticklabels(PARAM_LABELS if r == 1 else [""]*20, rotation=60, ha="right", fontsize=8)
            ax.legend(fontsize=8, loc="lower right", ncol=2); ax.set_ylabel("coefficient", fontsize=10)
        fig.tight_layout(); savef(fig, d, i)
    encode(d, OUT / "anim_fhn_phase.mp4", 1/1.4)

@anim
def anim_fhn_multiple_shooting():
    seg = read_csv("anim_concept_segments.csv"); cost = read_csv("anim_concept_costs.csv").set_index("window_size")
    dat = read_csv("fhn_data.csv"); fine = read_csv("fhn_fine.csv"); d = frames_dir("fhn_ms"); n = 0
    def base(title):
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(fine.t, fine.v, color=GREY, lw=0.9, ls="--", label="true trajectory"); ax.plot(dat.t, dat.v, ".", color=INK, ms=4, label="noisy data")
        ax.set_ylim(-3, 3); ax.set_xlim(-1, 101); ax.set_xlabel("time t", fontsize=11); ax.set_ylabel("v", fontsize=11); ax.set_title(title, fontsize=12); return fig, ax
    # part 1: single shooting drawn progressively (κ=100)
    g = seg[seg.window_size == 100]
    for tmax in np.linspace(2, 100, 50):
        fig, ax = base(r"single shooting ($\kappa=100$, one window): the model at a slightly wrong $p$ is integrated from $y_0$ over the whole record")
        gg = g[g.t <= tmax]; ax.plot(gg.t, gg.v, color=VERMILION, lw=2, label="model at wrong p")
        dd = dat[dat.t <= tmax]; mm = np.interp(dd.t, gg.t, gg.v); ax.vlines(dd.t, dd.v, mm, color=VERMILION, lw=0.6, alpha=0.6)
        ax.plot(dat.t.values[0], dat.v.values[0], "o", mfc="white", mec=VERMILION, ms=8, mew=1.5, label="shooting node")
        ax.legend(fontsize=9, loc="upper right", ncol=4); ax.text(0.01, 0.04, rf"$J_{{100}}(p_{{\rm wrong}})$ = {cost.J_wrong.loc[100]:.3g}   (true $p$: {cost.J_true.loc[100]:.3g})", transform=ax.transAxes, fontsize=10, color=VERMILION)
        fig.tight_layout(); savef(fig, d, n); n += 1
    for _ in range(12): shutil.copy(d / f"frame_{n-1:04d}.png", d / f"frame_{n:04d}.png"); n += 1
    # part 2: κ=10 windows appear one by one
    g = seg[seg.window_size == 10]
    for w in range(1, 11):
        fig, ax = base(r"multiple shooting ($\kappa=10$): each window restarts from the datum at its left node — the error cannot accumulate")
        for wi in range(1, w + 1):
            gg = g[g.window == wi]; ax.axvspan(gg.t.min(), gg.t.max(), color=BLUE if wi % 2 else LIGHTGREY, alpha=0.08)
            ax.plot(gg.t, gg.v, color=VERMILION, lw=2); node = gg[gg.is_node]; ax.plot(node.t, node.v, "o", mfc="white", mec=VERMILION, ms=8, mew=1.5)
            dd = dat[(dat.t >= gg.t.min()) & (dat.t <= gg.t.max())]; ax.vlines(dd.t, dd.v, np.interp(dd.t, gg.t, gg.v), color=VERMILION, lw=0.6, alpha=0.6)
        ax.text(0.01, 0.04, rf"$J_{{10}}(p_{{\rm wrong}})$ = {cost.J_wrong.loc[10]:.3g}   (true $p$: {cost.J_true.loc[10]:.3g})", transform=ax.transAxes, fontsize=10, color=VERMILION)
        fig.tight_layout(); savef(fig, d, n); n += 1
        for _ in range(4): shutil.copy(d / f"frame_{n-1:04d}.png", d / f"frame_{n:04d}.png"); n += 1
    for _ in range(12): shutil.copy(d / f"frame_{n-1:04d}.png", d / f"frame_{n:04d}.png"); n += 1
    # part 3: node removal — κ sweeps upward
    for k in FULL:
        fig, ax = base(rf"removing nodes: $\kappa$ = {k}  ({int(np.ceil(100/k))} windows) — the same wrong $p$ is penalised more as windows grow")
        g = seg[seg.window_size == k]
        for wi, gg in g.groupby("window"):
            ax.axvspan(gg.t.min(), gg.t.max(), color=BLUE if wi % 2 else LIGHTGREY, alpha=0.08); ax.plot(gg.t, gg.v, color=VERMILION, lw=1.8)
            node = gg[gg.is_node]; ax.plot(node.t, node.v, "o", mfc="white", mec=VERMILION, ms=6, mew=1.2)
        ax.text(0.01, 0.04, rf"$J_\kappa(p_{{\rm wrong}})$ = {cost.J_wrong.loc[k]:.3g}   (true $p$: {cost.J_true.loc[k]:.3g})   ratio {cost.J_wrong.loc[k]/cost.J_true.loc[k]:.1f}", transform=ax.transAxes, fontsize=10, color=VERMILION)
        fig.tight_layout(); savef(fig, d, n); n += 1
        for _ in range(11): shutil.copy(d / f"frame_{n-1:04d}.png", d / f"frame_{n:04d}.png"); n += 1
    encode(d, OUT / "anim_fhn_multiple_shooting.mp4", 12)

@anim
def anim_fhn_landscape():
    L = read_csv("anim_landscape_v_v3.csv"); _, _, sw, mins = _fhn_common(); d = frames_dir("fhn_landscape")
    vmin = L.J[L.J < 1e3].min(); vmax = L.J[L.J < 1e3].quantile(0.98)
    path = {arm: mins[(mins.arm == arm) & (mins["index"].isin([7, 12]))].pivot(index="window_size", columns="index", values="value") for arm in ("propagate", "reset")}  # index 7 = p_v (z index 5+2), 12 = p_v3
    for i, k in enumerate(FULL):
        fig, ax = plt.subplots(figsize=(7.5, 5.6))
        g = L[L.window_size == k]; X = np.sort(g.dp_v.unique()) + 1.0; Y = np.sort(g.dp_v3.unique()) - 1/3
        Z = g.pivot(index="dp_v3", columns="dp_v", values="J").values
        im = ax.pcolormesh(X, Y, np.ma.masked_where(Z >= 1e3, Z), cmap=SEQ_CMAP + "_r", norm=LogNorm(vmin=vmin, vmax=vmax), shading="nearest")
        ax.pcolormesh(X, Y, np.ma.masked_where(Z < 1e3, np.ones_like(Z)), cmap="Greys", vmin=0, vmax=1.6, shading="nearest")
        ax.contour(X, Y, np.ma.masked_where(Z >= 1e3, Z), levels=np.geomspace(vmin, vmax, 8), colors=GREY, linewidths=0.4)
        ax.plot(1.0, -1/3, marker="*", color=YELLOW, ms=14, mec=INK, mew=0.8, label=r"truth $p^\star$")
        for arm in ("propagate", "reset"):
            P = path[arm]; P = P[P.index <= k]
            ax.plot(P[7], P[12], "-o", color=ARM[arm], ms=4, lw=1.4, alpha=0.9, label=ARM_LABEL[arm] + " (seed 2)"); ax.plot(P[7].iloc[-1], P[12].iloc[-1], "o", color=ARM[arm], ms=9, mec="white", mew=1)
        ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_xlabel(r"coefficient of $v$ in $\dot v$", fontsize=11); ax.set_ylabel(r"coefficient of $v^3$ in $\dot v$", fontsize=11)
        ax.set_title(rf"cost landscape $J_\kappa$ at $\kappa$ = {k} (other coefficients at truth); grey = blow-up plateau", fontsize=11); ax.legend(fontsize=9, loc="lower left")
        fig.colorbar(im, ax=ax, pad=0.02, label=r"$J_\kappa$"); fig.tight_layout(); savef(fig, d, i)
    encode(d, OUT / "anim_fhn_landscape.mp4", 1/1.4)

def _lv_common():
    f = read_csv("anim_lv_fits.csv"); f = f[f.seed == 1]; dat = read_csv("lv_data.csv"); fine = read_csv("lv_fine.csv")
    o = read_csv("other_sweeps.csv"); o = o[(o.system == "lv") & (o.seed == 1)]
    mins = read_csv("other_minimizers.csv"); mins = mins[(mins.system == "lv") & (mins.seed == 1)]
    return f, dat, fine, o, mins, sorted(o.window_size.unique())

@anim
def anim_lv_sweep():
    f, dat, fine, o, mins, ks = _lv_common(); d = frames_dir("lv_sweep")
    for i, k in enumerate(ks):
        fig = plt.figure(figsize=(10, 4.6)); gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.7, 1])
        for r, arm in enumerate(("propagate", "reset")):
            ax = fig.add_subplot(gs[r, 0])
            for st, c in (("x", BLUE), ("y", ORANGE)):
                ax.plot(fine.t, fine[st], color=c, lw=0.9, ls="--", alpha=0.6); ax.plot(dat.t, dat[st], ".", color=c, ms=3)
                g = f[(f.arm == arm) & (f.window_size == k)]; ax.plot(g.t, g[st], color=c, lw=1.9, label=f"{st}(t) fit")
            nodes = dat.t.values[::k]; ax.plot(nodes, np.full_like(nodes, -0.4), "|", color=ARM[arm], ms=7, mew=1.3)
            ax.set_ylim(-0.6, 8); ax.set_ylabel(ARM_LABEL[arm], color=ARM[arm], fontsize=10); ax.legend(fontsize=9, loc="upper right", ncol=2)
            row = o[(o.arm == arm) & (o.window_size == k)]
            ax.text(0.01, 0.9, rf"$J_\kappa$ = {row.J.iloc[0]:.3g}    $\|p-p^\star\|$ = {row.p_err.iloc[0]:.2f}" + ("   (blow-up plateau)" if row.blowup.iloc[0] else ""), transform=ax.transAxes, fontsize=10, color=ARM[arm])
            if r == 0: ax.set_title(rf"Lotka–Volterra (dashed: truth, dots: data), window size $\kappa$ = {k}", fontsize=12)
            else: ax.set_xlabel("t", fontsize=11)
        for r, col in enumerate(("J", "p_err")):
            ax = fig.add_subplot(gs[r, 1])
            for arm in ("propagate", "reset"):
                g = o[o.arm == arm].sort_values("window_size"); gg = g[g.window_size <= k]
                ax.plot(g.window_size, g[col], color=ARM[arm], alpha=0.2, lw=1); ax.plot(gg.window_size, gg[col], color=ARM[arm], marker="o", ms=4, lw=1.8, label=ARM_LABEL[arm])
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel(r"cost $J_\kappa$" if col == "J" else r"$\|p^{(\kappa)}-p^\star\|$", fontsize=11); ax.axvline(k, color=LIGHTGREY, lw=0.8)
            if r == 0: ax.legend(fontsize=8)
            if r == 1: ax.set_xlabel(r"$\kappa$", fontsize=11)
        fig.tight_layout(); savef(fig, d, i)
    encode(d, OUT / "anim_lv_sweep.mp4", 1/1.6)

@anim
def anim_lv_phase():
    f, dat, fine, o, mins, ks = _lv_common(); d = frames_dir("lv_phase")
    for i, k in enumerate(ks):
        fig = plt.figure(figsize=(10, 4.6)); gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1, 1.4])
        ax = fig.add_subplot(gs[:, 0])
        ax.plot(fine.x, fine.y, color=GREY, lw=0.9, ls="--", label="true orbit"); ax.plot(dat.x, dat.y, ".", color=GREY, ms=3, label="data")
        for arm in ("propagate", "reset"):
            g = f[(f.arm == arm) & (f.window_size == k)]; ax.plot(g.x, g.y, color=ARM[arm], lw=1.6, label=ARM_LABEL[arm])
        ax.set_xlim(-0.3, 8); ax.set_ylim(-0.3, 6); ax.set_xlabel("x (prey)", fontsize=11); ax.set_ylabel("y (predator)", fontsize=11); ax.legend(fontsize=9, loc="upper right")
        ax.set_title(rf"Lotka–Volterra phase plane, minimiser at $\kappa$ = {k}", fontsize=11)
        for r, arm in enumerate(("propagate", "reset")):
            ax = fig.add_subplot(gs[r, 1]); x = np.arange(12)
            ax.bar(x - 0.2, LV_P, width=0.4, color=GREY, label=r"truth $p^\star$")
            g = mins[(mins.arm == arm) & (mins.window_size == k) & (mins["index"] > 2)].sort_values("index")
            ax.bar(x + 0.2, g.value.values, width=0.4, color=ARM[arm], label=ARM_LABEL[arm])
            row = o[(o.arm == arm) & (o.window_size == k)]
            ax.text(0.99, 0.95, rf"$\|p-p^\star\|$ = {row.p_err.iloc[0]:.2f}   score {row.score.iloc[0]:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=10, color=ARM[arm])
            ax.axhline(0, color=GREY, lw=0.6); ax.set_ylim(-1.3, 1.6); ax.set_xticks(x); ax.set_xticklabels(LV_LABELS if r == 1 else [""]*12, rotation=45, ha="right", fontsize=9)
            ax.legend(fontsize=8, loc="lower right", ncol=2); ax.set_ylabel("coefficient", fontsize=10)
        fig.tight_layout(); savef(fig, d, i)
    encode(d, OUT / "anim_lv_phase.mp4", 1/1.6)


LV_PLANES = {  # name: (csv suffix, col a, col b, z index a, z index b, truth a, truth b, label a, label b)
    "x_xy":  ("x_xy",  "dp_x",    "dp_xy",   6, 7,  1.0, -0.5, r"coefficient of $x$ in $\dot x$ (true 1.0)",   r"coefficient of $xy$ in $\dot x$ (true $-0.5$)"),
    "x2_xy": ("x2_xy", "dp_x2",   "dp_xy",   8, 7,  0.0, -0.5, r"coefficient of $x^2$ in $\dot x$ (true 0)",  r"coefficient of $xy$ in $\dot x$ (true $-0.5$)"),
    "xy_xy": ("xy_xy", "dp_xy_x", "dp_xy_y", 7, 13, -0.5, 0.3, r"coefficient of $xy$ in $\dot x$ (true $-0.5$)", r"coefficient of $xy$ in $\dot y$ (true 0.3)"),
}
def _lv_landscape(plane, out_name):
    """Lotka–Volterra: 2-D cost landscape in one coefficient plane at every FULL window size,
    with the GP and control paths (seed 1; the LV sweep visits κ ∈ SHORT) overlaid."""
    suf, ca, cb, ia, ib, ta, tb, la, lb = LV_PLANES[plane]
    L = read_csv(f"anim_lv_landscape_{suf}.csv"); _, _, _, o, mins, ks = _lv_common(); d = frames_dir(out_name)
    vmin = L.J[L.J < 1e3].min(); vmax = L.J[L.J < 1e3].quantile(0.98)
    path = {arm: mins[(mins.arm == arm) & (mins["index"].isin([ia, ib]))].pivot(index="window_size", columns="index", values="value") for arm in ("propagate", "reset")}
    for i, k in enumerate(FULL):
        fig, ax = plt.subplots(figsize=(7.5, 5.6))
        g = L[L.window_size == k]; X = np.sort(g[ca].unique()) + ta; Y = np.sort(g[cb].unique()) + tb
        Z = g.pivot(index=cb, columns=ca, values="J").values
        im = ax.pcolormesh(X, Y, np.ma.masked_where(Z >= 1e3, Z), cmap=SEQ_CMAP + "_r", norm=LogNorm(vmin=vmin, vmax=vmax), shading="nearest")
        ax.pcolormesh(X, Y, np.ma.masked_where(Z < 1e3, np.ones_like(Z)), cmap="Greys", vmin=0, vmax=1.6, shading="nearest")
        ax.contour(X, Y, np.ma.masked_where(Z >= 1e3, Z), levels=np.geomspace(vmin, vmax, 10), colors=GREY, linewidths=0.35)
        ax.plot(ta, tb, marker="*", color=YELLOW, ms=14, mec=INK, mew=0.8, label=r"truth $p^\star$")
        for arm in ("propagate", "reset"):
            P = path[arm]; P = P[P.index <= k]
            if len(P):
                ax.plot(P[ia], P[ib], "-o", color=ARM[arm], ms=4, lw=1.4, alpha=0.9, label=ARM_LABEL[arm] + " (seed 1)"); ax.plot(P[ia].iloc[-1], P[ib].iloc[-1], "o", color=ARM[arm], ms=9, mec="white", mew=1)
        ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_xlabel(la, fontsize=11); ax.set_ylabel(lb, fontsize=11)
        ax.set_title(rf"Lotka–Volterra cost landscape $J_\kappa$ at $\kappa$ = {k} (other coefficients at truth); grey = blow-up", fontsize=11); ax.legend(fontsize=9, loc="lower left")
        fig.colorbar(im, ax=ax, pad=0.02, label=r"$J_\kappa$"); fig.tight_layout(); savef(fig, d, i)
    encode(d, OUT / f"{out_name}.mp4", 1/1.4)

@anim
def anim_lv_landscape(): _lv_landscape("x_xy", "anim_lv_landscape")
@anim
def anim_lv_landscape_x2_xy(): _lv_landscape("x2_xy", "anim_lv_landscape_x2_xy")
@anim
def anim_lv_landscape_xy_xy(): _lv_landscape("xy_xy", "anim_lv_landscape_xy_xy")

if __name__ == "__main__":
    for n in (sys.argv[1:] or list(ANIMS)):
        ANIMS[n]()
