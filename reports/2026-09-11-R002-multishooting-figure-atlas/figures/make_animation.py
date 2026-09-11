"""make_animation.py — animated GIF of the guess-propagation sweep (optimiser seed 2):
left: the single-shooting simulation from the current minimiser vs data; right: the cost
and parameter error so far. One frame per window size, GP (blue) and control (vermilion).
Writes figures/anim_sweep.gif (+ the frames as PNG in figures/anim_frames/). PIL only; no ffmpeg.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from style import *
import pandas as pd
from PIL import Image
from matplotlib import gridspec

f = read_csv("sweep_fits.csv"); dat = read_csv("fhn_data.csv"); fine = read_csv("fhn_fine.csv"); sw = read_csv("sweeps.csv")
seed = 2 if (f.seed == 2).any() else 1
sw = sw[(sw.exp == "main") & (sw.seed == seed)]
ks = sorted(f.window_size.unique())
frames_dir = OUT / "anim_frames"; frames_dir.mkdir(exist_ok=True)
paths = []
for i, k in enumerate(ks):
    fig = plt.figure(figsize=(9, 4.2), dpi=100); gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.6, 1])
    for r, (arm, st) in enumerate((("propagate", "v"), ("reset", "v"))):
        ax = fig.add_subplot(gs[r, 0])
        ax.plot(fine.t, fine.v, color=GREY, lw=0.9, ls="--", label="truth"); ax.plot(dat.t, dat.v, ".", color=GREY, ms=3, label="data")
        g = f[(f.arm == arm) & (f.window_size == k) & (f.seed == seed)]
        ax.plot(g.t, g.v, color=ARM[arm], lw=1.6, label=ARM_LABEL[arm])
        # shooting nodes of this window size
        nodes = dat.t.values[::k]
        ax.plot(nodes, np.full_like(nodes, -2.8), "|", color=ARM[arm], ms=6, mew=1.2)
        ax.set_ylim(-3, 3); ax.set_ylabel("v(t)"); ax.legend(fontsize=7, loc="upper right", ncol=3)
        row = sw[(sw.arm == arm) & (sw.window_size == k)]
        if len(row):
            ax.text(0.01, 0.04, rf"$J_\kappa$ = {row.J.iloc[0]:.4g}   $\|p-p^\star\|$ = {row.p_err.iloc[0]:.3f}" + ("   (blow-up plateau)" if row.blowup.iloc[0] else ""),
                    transform=ax.transAxes, fontsize=8, color=ARM[arm])
        if r == 0: ax.set_title(rf"window size $\kappa$ = {k}  ({int(np.ceil(100/k))} windows; ticks = shooting nodes)", fontsize=10)
        else: ax.set_xlabel("t")
    for r, col in enumerate(("J", "p_err")):
        ax = fig.add_subplot(gs[r, 1])
        for arm in ("propagate", "reset"):
            g = sw[(sw.arm == arm)].sort_values("window_size"); gg = g[g.window_size <= k]
            ax.plot(g.window_size, g[col], color=ARM[arm], alpha=0.2, lw=1); ax.plot(gg.window_size, gg[col], color=ARM[arm], marker="o", ms=3, lw=1.6)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel(r"$J_\kappa$" if col == "J" else r"$\|p^{(\kappa)}-p^\star\|$")
        if r == 1: ax.set_xlabel(r"$\kappa$")
        ax.axvline(k, color=LIGHTGREY, lw=0.8)
    fig.tight_layout()
    pth = frames_dir / f"frame_{i:02d}.png"; fig.savefig(pth, dpi=100); plt.close(fig); paths.append(pth)
imgs = [Image.open(p).convert("P", palette=Image.ADAPTIVE) for p in paths]
imgs[0].save(OUT / "anim_sweep.gif", save_all=True, append_images=imgs[1:] + [imgs[-1]]*3, duration=700, loop=0)
print("wrote anim_sweep.gif with", len(imgs), "frames (seed", seed, ")")
