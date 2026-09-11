"""style.py — shared matplotlib style for every R002 figure (import first).

Palette: Okabe–Ito (colour-blind safe, validated with the dataviz palette checker,
adjacent-pair CVD ΔE ≥ 7.6 with direct labels/markers as secondary encoding).
Categorical hues are assigned in FIXED order by entity (never cycled):
    ARM colours  : propagate=blue, reset=vermilion, best=green, graded=purple
    STATE colours: v=blue, w=orange ; truth=grey dashed ; data=dark grey dots
Every figure is saved as PDF (for LaTeX) + PNG (preview) via `save(fig, name)`.
"""
import os, pathlib
os.environ.setdefault("MPLCONFIGDIR", os.environ.get("TMPDIR", "/tmp") + "/mplcfg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "analysis" / "results"
OUT = HERE                       # figures land next to the scripts

BLUE, VERMILION, GREEN, PURPLE, ORANGE, SKY, YELLOW, BLACK = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000")
GREY, LIGHTGREY, INK, MUTED = "#4D4D4D", "#BBBBBB", "#222222", "#777777"

ARM = {"propagate": BLUE, "reset": VERMILION, "best": GREEN,
       "propagate_graded": PURPLE, "reset_graded": ORANGE,
       "bfgs_propagate": GREEN, "lbfgs_propagate": PURPLE, "bfgs_reset": ORANGE, "lbfgs_reset": SKY}
ARM_LABEL = {"propagate": "guess propagation (GP)", "reset": "no propagation (control)",
             "best": "best-guess propagation", "propagate_graded": "GP + graded penalty",
             "reset_graded": "control + graded penalty",
             "bfgs_propagate": "BFGS + GP", "lbfgs_propagate": "L-BFGS + GP",
             "bfgs_reset": "BFGS, no propagation", "lbfgs_reset": "L-BFGS, no propagation"}
ARM_MARKER = {"propagate": "o", "reset": "s", "best": "^", "propagate_graded": "D", "reset_graded": "v",
              "bfgs_propagate": "P", "lbfgs_propagate": "X", "bfgs_reset": "p", "lbfgs_reset": "h"}
STATE = {"v": BLUE, "w": ORANGE, "x": BLUE, "y": ORANGE, "z": GREEN}
SEQ_CMAP = "Blues"          # sequential = one hue light→dark
DIV_CMAP = "RdBu_r"         # diverging = two hues + neutral midpoint

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY,
    "axes.grid": False, "lines.linewidth": 1.6, "lines.markersize": 4.5,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200,
    "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.fontset": "cm",
})

def save(fig, name, tight=True):
    """Save `name`.pdf + `name`.png into figures/ and close."""
    if tight:
        fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print("saved", name)

def read_csv(name):
    import pandas as pd
    return pd.read_csv(RES / name)

def read_json(name):
    import json
    return json.loads((RES / name).read_text())

def label_end(ax, x, y, text, color, dx=0.01, **kw):
    """Direct label at the end of a line (secondary encoding besides colour)."""
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", color=color,
                va="center", ha="left", fontsize=8, **kw)

def panel_label(ax, s, x=-0.12, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="left", color=INK)

MONO_LABELS = ["1", "w", "w²", "w³", "v", "vw", "vw²", "v²", "v²w", "v³"]     # library order
PARAM_LABELS = [f"v̇: {m}" for m in MONO_LABELS] + [f"ẇ: {m}" for m in MONO_LABELS]
