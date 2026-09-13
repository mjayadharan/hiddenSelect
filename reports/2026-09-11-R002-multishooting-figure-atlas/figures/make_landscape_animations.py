"""make_landscape_animations.py — high-resolution 2-D and 3-D cost-landscape animations.

Extends the landscape animation of `make_animations.py` (`anim_fhn_landscape`, 61x61, 825x616 px)
in three ways:

  (i)   MORE PLANES. Five FitzHugh-Nagumo coefficient planes instead of one, chosen by the
        screen in `analysis/12_landscape_planes.jl` for how drastically the landscape changes
        with the window size kappa (report section 7.1), not by hand.
  (ii)  MORE RESOLUTION. 201x201 cost grids (10.9x the samples of the original FHN grid, 1.56x
        the 161x161 Lotka-Volterra grids) rendered at 2560x1440 (1440p) instead of 825x616.
  (iii) A 3-D VERSION. The same morphing surface z = log10 J_kappa drawn as a rotating 3-D
        surface with the blow-up plateau as a flat mesa, a contour shadow on the floor and the
        guess-propagation / control paths walked over the surface.

Outputs (figures/):
  anim_fhn_landscape_<plane>_hires.mp4     2-D, 1440p, one frame per kappa      (5 planes)
  anim_lv_landscape_x2_xy_hires.mp4        2-D, 1440p, Lotka-Volterra (x^2, xy)
  anim3d_fhn_landscape_<plane>.mp4         3-D rotating surface                 (3 planes)
  anim3d_lv_landscape_x2_xy.mp4            3-D rotating surface, Lotka-Volterra

Plane geometry (indices, truths, axis labels) is read from
`analysis/results/landscape_planes_meta.json` and `landscape_lv_hires_meta.json`, both written by
the Julia stage, so the two sides cannot drift apart.

Usage: python figures/make_landscape_animations.py [2d|3d|all] [plane ...]
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import *
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LogNorm, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3-D projection)
from make_animations import ffenv, FFMPEG
import subprocess, shutil

FULL = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 33, 50, 75, 100]
BLOW = 1e3                 # ms_loss returns exactly this on finite-time blow-up
DPI = 320                  # 8.0 x 4.5 in at 320 dpi = 2560 x 1440 (1440p)
FIGSIZE = (8.0, 4.5)
SUB = 16                   # 3-D frames per kappa (one full 360 deg turn over the animation)

def short(lab):
    """Axis label trimmed for the cramped 3-D axes."""
    return lab.replace("coefficient of ", "").replace("constant in ", "const. of ")

def nK(k):
    """Number of shooting windows (= number of nodes) for window size kappa: K = ceil(100/kappa).
    Both systems here have N = 101 samples, hence 100 data intervals."""
    return int(np.ceil(100 / k))

def nwin(k):
    n = nK(k)
    return f"{n} window" + ("s" if n > 1 else "")

# Blues_r truncated at 0.92 so the high-cost end never reaches pure white (invisible in 3-D)
CMAP = mcolors.LinearSegmentedColormap.from_list(
    "Blues_r_trunc", plt.get_cmap("Blues_r")(np.linspace(0.0, 0.92, 256)))
PLATEAU = "#9E9E9E"

# ------------------------------------------------------------------- io -------
def meta_all():
    m = json.loads((RES / "landscape_planes_meta.json").read_text())
    m.update({("lv_" + k): v for k, v in json.loads((RES / "landscape_lv_hires_meta.json").read_text()).items()})
    return m

def load_grid(m):
    """-> (kappas, X, Y, J) with J[k, ix, iy] on the ABSOLUTE coefficient axes X (a) and Y (b)."""
    n = m["n"]
    df = pd.read_csv(RES / m["file"])
    assert len(df) == n * n * len(FULL), f"{m['file']}: {len(df)} rows, expected {n*n*len(FULL)}"
    ks = df.window_size.values.reshape(len(FULL), n * n)[:, 0]
    assert list(ks) == FULL, "kappa ladder in the grid file does not match FULL"
    A = df[m["col_a"]].values[:n * n].reshape(n, n)[:, 0]      # offset varies along axis 0
    B = df[m["col_b"]].values[:n * n].reshape(n, n)[0, :]      # offset varies along axis 1
    J = df.J.values.reshape(len(FULL), n, n)
    return ks, A + m["truth_a"], B + m["truth_b"], J

def load_paths(m):
    """GP / control minimiser paths projected on the plane, keyed by arm -> DataFrame[a, b]."""
    if m["system"] == "fhn":
        mins = read_csv("sweep_minimizers.csv"); mins = mins[(mins.exp == "main") & (mins.seed == m["seed"])]
    else:
        mins = read_csv("other_minimizers.csv"); mins = mins[(mins.system == "lv") & (mins.seed == m["seed"])]
    ia, ib = m["z_index_a"], m["z_index_b"]
    out = {}
    for arm in ("propagate", "reset"):
        g = mins[(mins.arm == arm) & (mins["index"].isin([ia, ib]))]
        if not len(g):
            continue
        P = g.pivot(index="window_size", columns="index", values="value")
        out[arm] = P.rename(columns={ia: "a", ib: "b"})[["a", "b"]].sort_index()
    return out

def frames_dir(name):
    d = OUT / "anim_frames" / name
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True); return d

def savef(fig, d, i):
    fig.savefig(d / f"frame_{i:04d}.png", dpi=DPI, facecolor="white"); plt.close(fig)

def encode(d, out, in_fps, out_fps=24, crf=18):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-framerate", str(in_fps),
                    "-i", str(d / "frame_%04d.png"), "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuv420p",
                    "-r", str(out_fps), "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                   check=True, env=ffenv())
    print("wrote", out.name, f"({out.stat().st_size/1e6:.1f} MB)")

# --------------------------------------------------------------- geometry -----
GEOM = []      # per (plane, kappa) geometry at the production resolution, dumped for tables/

def scalars(J):
    """Per-kappa blow-up fraction and basin fraction (cells within 2x the plane minimum)."""
    fb, ba = [], []
    for k in range(J.shape[0]):
        Z = J[k]; fin = Z < BLOW
        fb.append(1 - fin.mean()); ba.append((Z < 2 * Z[fin].min()).mean())
    return np.array(fb), np.array(ba)

def limits(J):
    fin = J[J < BLOW]
    return fin.min(), np.quantile(fin, 0.995)

def surface_z(X, Y, J_k, a, b, lo, hi):
    """Height of the clipped log10 display surface at the nearest grid node to (a, b)."""
    return float(np.clip(np.log10(J_k[np.abs(X - a).argmin(), np.abs(Y - b).argmin()]), lo, hi))

# ------------------------------------------------------------------- 2-D ------
def render2d(key, m):
    ks, X, Y, J = load_grid(m); paths = load_paths(m)
    vmin, vmax = limits(J); fb, ba = scalars(J)
    for kk, k in enumerate(FULL):                       # the numbers the right-hand panels display
        fin = J[kk][J[kk] < BLOW]
        GEOM.append(dict(plane=key, n=m["n"], window_size=k, f_blow=fb[kk], basin=ba[kk],
                         J_min=fin.min(), J_max=fin.max()))
    norm = LogNorm(vmin=vmin, vmax=vmax); levels = np.geomspace(vmin, vmax, 11)
    sysname = "FitzHugh--Nagumo" if m["system"] == "fhn" else "Lotka--Volterra"
    d = frames_dir(f"ls2d_{key}")
    for i, k in enumerate(FULL):
        Z = J[i].T                                    # -> [iy, ix] for pcolormesh
        fig = plt.figure(figsize=FIGSIZE)
        gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[2.35, 1], hspace=0.42, wspace=0.28)
        ax = fig.add_subplot(gs[:, 0])
        fin = np.ma.masked_where(Z >= BLOW, Z)
        im = ax.pcolormesh(X, Y, fin, cmap=CMAP, norm=norm, shading="nearest", rasterized=True)
        ax.pcolormesh(X, Y, np.ma.masked_where(Z < BLOW, np.ones_like(Z)),
                      cmap=mcolors.ListedColormap([PLATEAU]), vmin=0, vmax=1, shading="nearest", rasterized=True)
        ax.contour(X, Y, fin, levels=levels, colors=GREY, linewidths=0.3, alpha=0.8)
        ax.plot(m["truth_a"], m["truth_b"], marker="*", color=YELLOW, ms=13, mec=INK, mew=0.7,
                label=r"truth $p^\star$", zorder=5)
        for arm, P in paths.items():
            Q = P[P.index <= k]
            if not len(Q):
                continue
            ax.plot(Q.a, Q.b, "-o", color=ARM[arm], ms=3, lw=1.2, alpha=0.95, zorder=4,
                    label=ARM_LABEL[arm] + f" (seed {m['seed']})")
            ax.plot(Q.a.iloc[-1], Q.b.iloc[-1], "o", color=ARM[arm], ms=7, mec="white", mew=1.0, zorder=6)
        ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max())
        ax.set_xlabel(m["label_a"], fontsize=8.5); ax.set_ylabel(m["label_b"], fontsize=8.5)
        ax.set_title(rf"{sysname.replace('--','–')} cost landscape $J_\kappa$, $\kappa$ = {k}"
                     f"  ({nwin(k)})", fontsize=9.5)
        ax.legend(fontsize=6.5, loc="lower left", ncol=1)
        cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.045); cb.set_label(r"$J_\kappa$", fontsize=8.5)
        cb.ax.tick_params(labelsize=7)
        # right column: how the plane's geometry evolves along the ladder
        for r, (vals, lab, col) in enumerate(((ba, "basin: area with\n" + r"$J<2\,J_{\min}$", BLUE),
                                              (fb, "blow-up plateau:\n" + r"area with $J\geq10^3$", PLATEAU))):
            a2 = fig.add_subplot(gs[r, 1])
            a2.plot(FULL, np.maximum(vals, 1e-5), color=col, lw=1.3, alpha=0.35)
            a2.plot(FULL[:i+1], np.maximum(vals[:i+1], 1e-5), color=col, lw=1.6, marker="o", ms=2.6)
            a2.plot(k, max(vals[i], 1e-5), "o", color=col, ms=5.5, mec="white", mew=0.9)
            a2.set_xscale("log")
            if r == 0:
                a2.set_yscale("log")
            else:
                a2.set_ylim(-0.03, 1.03)
            a2.set_ylabel(lab, fontsize=7); a2.tick_params(labelsize=7)
            a2.set_xlabel(r"window size $\kappa$", fontsize=7.5) if r == 1 else a2.set_xticklabels([])
            a2.text(0.97, 0.92, f"{vals[i]*100:.2g} % of the plane", transform=a2.transAxes,
                    ha="right", va="top", fontsize=7, color=col)
        fig.subplots_adjust(left=0.085, right=0.965, top=0.93, bottom=0.115)
        savef(fig, d, i)
    encode(d, OUT / f"{'anim_fhn_landscape_' + key if m['system']=='fhn' else 'anim_lv_landscape_x2_xy'}_hires.mp4", 1 / 1.5)

# ------------------------------------------------------------------- 3-D ------
# Display convention: the surface is z = log10 J_kappa CLIPPED at the 99.5th percentile of the
# finite costs on the plane. Without the clip the blow-up value (J = 1e3, three decades above the
# bulk of the plane) owns most of the vertical axis and flattens the valley that is the subject of
# the picture. Clipped cells are drawn grey and named in the title; no cost value is altered
# anywhere except in the height and colour of this one surface.
def render3d(key, m, mode="arms", elev=32, tag="", index="kappa", info=True, stop_K=None):
    """mode="arms": overlay the GP and control minimiser paths (the default).
    mode="control": ONLY the no-propagation arm -- the minimiser the optimiser returns at each
    kappa when every stage restarts from the same seed. This is what the method does BEFORE guess
    propagation is introduced.
    stop_K=n ends the animation at the FIRST stage whose window count K = ceil(100/kappa) reaches n,
    instead of running the ladder out to kappa = 100 (K = 1, single shooting). K is not injective on
    this ladder -- kappa = 50 and 75 both give K = 2 -- so stop_K=2 ends at kappa = 50 and drops
    kappa = 75 (a second K = 2) and kappa = 100.
    info=False drops the per-frame readout block in the left gutter (K/kappa, basin, plateau, J at
    p*), leaving the title to carry the stage. The title still prints both K and kappa.
    mode="minimum": no optimiser paths at all -- mark the minimum of the DRAWN plane, argmin
    J_kappa. Note "minimum" and "control" are different objects: argmin J_kappa is the best point of
    a slice whose other 18 coefficients are pinned at truth, so it sits on the truth star at every
    kappa; the control minimiser is a full 20-dimensional optimiser output projected down and does
    not."""
    ks, X, Y, J = load_grid(m)
    paths = load_paths(m) if mode in ("arms", "control") else {}
    assert index in ("kappa", "K")
    if mode == "control":
        paths = {"reset": paths["reset"]}
    vmin, vmax = limits(J); fb, ba = scalars(J)
    GX, GY = np.meshgrid(X, Y, indexing="ij")
    zfloor, zceil = np.log10(vmin), np.log10(vmax)
    zbase = zfloor - 0.30 * (zceil - zfloor)          # height of the projected contour floor
    norm = Normalize(vmin=zfloor, vmax=zceil)
    sysname = "FitzHugh–Nagumo" if m["system"] == "fhn" else "Lotka–Volterra"
    d = frames_dir(f"ls3d_{key}" + ("" if mode == "arms" else f"_{mode}") + tag)
    jmins = np.array([J[i][J[i] < BLOW].min() for i in range(len(FULL))])
    ladder = FULL if stop_K is None else FULL[:next(i for i, kk in enumerate(FULL) if nK(kk) <= stop_K) + 1]
    perr = None
    if mode == "control":      # full-dimensional parameter error of the no-propagation arm
        sw = read_csv("sweeps.csv" if m["system"] == "fhn" else "other_sweeps.csv")
        sw = sw[(sw.arm == "reset") & (sw.seed == m["seed"])]
        sw = sw[sw.exp == "main"] if m["system"] == "fhn" else sw[sw.system == "lv"]
        perr = np.array([float(sw[sw.window_size == k].p_err.iloc[0]) for k in FULL])
        seed_perr = float(sw.seed_p_err.iloc[0])   # the shared starting guess, NOT the kappa=1 result
    n_hold = [2 * SUB] + [SUB] * (len(ladder) - 2) + [2 * SUB]
    total = sum(n_hold); f = 0
    for i, k in enumerate(ladder):
        Zc = np.log10(np.clip(J[i], vmin, vmax))       # clipped display surface
        blown = J[i] >= BLOW
        fc = CMAP(norm(Zc)); fc[blown] = mcolors.to_rgba(PLATEAU)
        it = (np.abs(X - m["truth_a"]).argmin(), np.abs(Y - m["truth_b"]).argmin())
        zt = float(np.clip(np.log10(J[i][it]), zfloor, zceil))
        ja, jb = np.unravel_index(np.argmin(np.where(J[i] < BLOW, J[i], np.inf)), J[i].shape)
        amin = (float(X[ja]), float(Y[jb]), float(J[i][ja, jb]))
        for s in range(n_hold[i]):
            fig = plt.figure(figsize=FIGSIZE)
            ax = fig.add_axes([0.115, 0.005, 0.885, 0.925], projection="3d", computed_zorder=False)
            ax.plot_surface(GX, GY, Zc, facecolors=fc, rstride=1, cstride=1, shade=False,
                            linewidth=0, antialiased=False, zorder=2)
            ax.contourf(X, Y, Zc.T, zdir="z", offset=zbase, levels=24, cmap=CMAP, norm=norm,
                        alpha=0.7, zorder=1)
            # truth: a star on the surface, its shadow on the floor, a stem joining them
            ax.plot([m["truth_a"]] * 2, [m["truth_b"]] * 2, [zbase, zt], color=INK, lw=0.7,
                    alpha=0.55, zorder=3)
            ax.plot([m["truth_a"]], [m["truth_b"]], [zbase], marker="*", color=YELLOW, ms=8,
                    mec=INK, mew=0.5, alpha=0.55, zorder=3)
            ax.plot([m["truth_a"]], [m["truth_b"]], [zt], marker="*", color=YELLOW, ms=15,
                    mec=INK, mew=0.8, ls="none", label=r"truth $p^\star$", zorder=8)
            if mode == "minimum":       # the minimum OF THE PLANE, not an optimiser iterate
                za = float(np.clip(np.log10(amin[2]), zfloor, zceil))
                ax.plot([amin[0]] * 2, [amin[1]] * 2, [zbase, za], color=VERMILION, lw=0.8,
                        alpha=0.7, zorder=4)
                ax.plot([amin[0]], [amin[1]], [zbase], marker="D", color=VERMILION, ms=4,
                        mec=INK, mew=0.4, alpha=0.5, zorder=3)
                ax.plot([amin[0]], [amin[1]], [za + 0.02 * (zceil - zfloor)], marker="D",
                        color=VERMILION, ms=7, mec="white", mew=0.9, ls="none", zorder=9,
                        label=r"plane minimum $\arg\min_\Pi J_\kappa$")
            for arm, P in paths.items():
                Q = P[P.index <= k]
                if not len(Q):
                    continue
                zs = [surface_z(X, Y, J[i], a, b, zfloor, zceil) + 0.02 * (zceil - zfloor)
                      for a, b in zip(Q.a, Q.b)]
                ax.plot(Q.a, Q.b, [zbase] * len(Q), "-", color=ARM[arm], lw=0.8, alpha=0.4, zorder=1)
                lab = (rf"minimiser at each $K$, no guess propagation (seed {m['seed']})"
                       if mode == "control" else ARM_LABEL[arm] + f" (seed {m['seed']})")
                ax.plot(Q.a, Q.b, zs, "-o", color=ARM[arm], ms=2.8, lw=1.3, zorder=6, label=lab)
                ax.plot([Q.a.iloc[-1]], [Q.b.iloc[-1]], [zs[-1]], "o", color=ARM[arm], ms=6.5,
                        mec="white", mew=0.9, ls="none", zorder=9)
            ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(zbase, zceil)
            ax.set_box_aspect((1.38, 1.18, 0.74), zoom=1.02 if elev < 45 else 0.90)
            ax.view_init(elev=elev, azim=-60 + 360.0 * f / total)
            ax.set_xlabel(short(m["label_a"]), fontsize=7.5, labelpad=2)
            ax.set_ylabel(short(m["label_b"]), fontsize=7.5, labelpad=2)
            ax.set_zlabel((r"$\log_{10} J$ (clipped)" if index == "K"
                           else r"$\log_{10} J_\kappa$ (clipped)"), fontsize=7.5, labelpad=-4)
            ax.tick_params(labelsize=6, pad=(2 if elev >= 45 else -1))
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_facecolor("white"); axis.pane.set_alpha(1.0)
                axis._axinfo["grid"]["color"] = "#EAEAEA"
            title = (rf"{sysname} cost landscape at $K$ = {nK(k)} {'windows' if nK(k) > 1 else 'window'}"
                     rf"   (window size $\kappa$ = {k})" if index == "K"
                     else rf"{sysname} cost landscape $J_\kappa$ at $\kappa$ = {k}  ({nwin(k)})")
            fig.suptitle(title, fontsize=10.5, y=0.975)
            jsym = "J" if index == "K" else r"J_\kappa"
            fig.text(0.5, 0.925, rf"surface $=\log_{{10}}{jsym}$ clipped at the 99.5th percentile of the "
                                 rf"finite costs; grey mesa $=$ blow-up plateau (${jsym}\geq10^3$)",
                     ha="center", fontsize=7, color=MUTED)
            h, lb = ax.get_legend_handles_labels()
            fig.legend(h, lb, fontsize=7, loc="lower left", bbox_to_anchor=(0.005, 0.03), frameon=False)
            if info:
                if index == "K":
                    lines = [rf"$K$ = {nK(k)} {'windows' if nK(k) > 1 else 'window'}  (nodes)",
                             rf"window size $\kappa$ = {k}",
                             rf"basin ($J<2J_{{\min}}$): {ba[i]*100:.3g} % of the plane",
                             rf"blow-up plateau: {fb[i]*100:.3g} % of the plane",
                             rf"$J$ at $p^\star$ = {J[i][it]:.4g}"]
                    if mode == "control":
                        lines += [rf"$\|p^{{(K)}}-p^\star\|$ = {perr[i]:.3f}   "
                                  rf"(starting guess: {seed_perr:.3f})"]
                else:
                    lines = [rf"$\kappa$ = {k}   ({nwin(k)})",
                             rf"basin ($J<2J_\min$): {ba[i]*100:.3g} % of the plane",
                             rf"blow-up plateau: {fb[i]*100:.3g} % of the plane",
                             rf"$J_\kappa^{{\min}}$ = {J[i][J[i] < BLOW].min():.4g}",
                             rf"$J_\kappa$ at $p^\star$ = {J[i][it]:.4g}"]
                    if mode == "minimum":
                        lines += [rf"$\arg\min_\Pi J_\kappa$ = ({amin[0]:.3f}, {amin[1]:.3f})",
                                  rf"distance to $p^\star$ = {np.hypot(amin[0]-m['truth_a'], amin[1]-m['truth_b']):.3f}"]
                fig.text(0.005, 0.885, "\n".join(lines),
                         fontsize=7.5, va="top", ha="left", color=INK, linespacing=1.9)
            if mode == "minimum":
                a2 = fig.add_axes([0.080, 0.345, 0.092, 0.205])
                yv = jmins[:len(ladder)]
                a2.plot(ladder, yv, color=VERMILION, lw=1.1, alpha=0.3)
                a2.plot(ladder[:i+1], yv[:i+1], color=VERMILION, lw=1.5, marker="o", ms=2.2)
                a2.plot(k, yv[i], "o", color=VERMILION, ms=5.5, mec="white", mew=0.9)
                a2.set_xscale("log"); a2.set_yscale("log")
                a2.set_ylim(yv.min() * 0.8, yv.max() * 1.25)
                a2.set_yticks([float(f"{v:.2g}") for v in np.geomspace(yv.min(), yv.max(), 3)])
                a2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
                a2.yaxis.set_minor_formatter(plt.NullFormatter())
                a2.tick_params(labelsize=5.5, pad=1)
                a2.set_ylabel(r"$J_\kappa^{\min}$", fontsize=7.5, labelpad=1)
                a2.set_xlabel(r"window size $\kappa$", fontsize=6, labelpad=1)
                cax = fig.add_axes([0.080, 0.245, 0.092, 0.011])     # horizontal, under the panel
                orient = "horizontal"
            else:
                cax = fig.add_axes([0.032, 0.24, 0.012, 0.29]); orient = "vertical"
            sm = ScalarMappable(norm=norm, cmap=CMAP); sm.set_array([])
            cb = fig.colorbar(sm, cax=cax, orientation=orient)
            cb.set_label(r"$\log_{10} J$" if index == "K" else r"$\log_{10} J_\kappa$",
                         fontsize=7); cb.ax.tick_params(labelsize=6)
            if orient == "vertical":
                cax.yaxis.set_ticks_position("left"); cax.yaxis.set_label_position("left")
            savef(fig, d, f); f += 1
        print(f"  {key} 3-D: kappa={k} ({f}/{total} frames)", flush=True)
    encode(d, OUT / (f"anim3d_{'fhn' if m['system']=='fhn' else 'lv'}_landscape_{key.replace('lv_','')}"
                     + ("" if mode == "arms" else f"_{mode}") + tag + ".mp4"), 24)

# ------------------------------------------------------------------ main ------

PLANES_3D = ["wv_ww", "v2_w2", "v_v3", "lv_x2_xy"]      # the drastic FHN planes + the LV (x^2, xy) plane

if __name__ == "__main__":
    args = sys.argv[1:]
    what = args[0] if args and args[0] in ("2d", "3d", "3d-min", "3d-control", "3d-arms-low", "all") else "all"
    rest = [a for a in args if a not in ("2d", "3d", "3d-min", "3d-control", "3d-arms-low", "all")]
    M = meta_all()
    if what in ("2d", "all"):
        for key in (rest or list(M)):
            print("2-D:", key, flush=True); render2d(key, M[key])
    if what in ("3d", "all"):
        for key in (rest or PLANES_3D):
            print("3-D:", key, flush=True); render3d(key, M[key])
    # camera elevations for the control cut: grazing (canyon profile), the default oblique view,
    # and near-overhead (the floor contour projection fully visible)
    CONTROL_VIEWS = [("_low", 14), ("", 32), ("_high", 62)]
    if what == "3d-min":
        for key in (rest or ["wv_ww"]):
            print("3-D (plane minimum, no optimiser paths):", key, flush=True)
            render3d(key, M[key], mode="minimum")
    if what == "3d-arms-low":      # both arms, grazing camera, indexed by K (the 2-D hires content)
        for key in (rest or ["wv_ww"]):
            print("3-D (both arms, elev 14, K-indexed):", key, flush=True)
            render3d(key, M[key], mode="arms", elev=14, tag="_low", index="K", info=False, stop_K=2)
    if what == "3d-control":
        views = CONTROL_VIEWS
        if rest and rest[-1] in [t.lstrip("_") or "mid" for t, _ in CONTROL_VIEWS]:
            sel = rest.pop()
            views = [v for v in CONTROL_VIEWS if (v[0].lstrip("_") or "mid") == sel]
        for key in (rest or ["wv_ww"]):
            for tag, elev in views:
                print(f"3-D (no-propagation arm only, elev {elev}):", key, flush=True)
                render3d(key, M[key], mode="control", elev=elev, tag=tag, index="K")
    if GEOM:
        # the basin/plateau numbers quoted in the report come from THIS grid, not the 61x61 screen
        g = pd.DataFrame(GEOM).drop_duplicates(["plane", "window_size"])
        f = RES / "landscape_hires_geometry.csv"
        if f.exists() and set(rest or []) :
            g = pd.concat([pd.read_csv(f), g]).drop_duplicates(["plane", "window_size"], keep="last")
        g.sort_values(["plane", "window_size"]).to_csv(f, index=False)
        print("wrote", f.name, len(g), "rows")
