"""sources.py -- where every inherited file of R004 comes from (shared by make_inputs_md.py and verify_R004.py)."""
from pathlib import Path
HERE = Path(__file__).resolve().parent; RPT = HERE.parent; REPORTS = RPT.parent
R002 = REPORTS / "2026-09-11-R002-multishooting-figure-atlas"; R003 = REPORTS / "2026-09-11-R003-tighter-error-bounds"
def source_of(rel: str):
    """rel = path relative to the R004 folder -> (source Path, producing note) or None."""
    name = Path(rel).name
    if rel.startswith("figures/R002_"): return R002 / "figures" / name[len("R002_"):], "R002 figures/make_figures.py"
    if rel.startswith("figures/R003_"): return R003 / "figures" / name[len("R003_"):], "R003 figures/make_figures.py"
    if rel.startswith("external_data/tables/"):
        if name == "R003_summary.json": return R003 / "analysis" / "results" / "summary.json", "R003 analysis/compute_R003.py"
        if name in ("lemmas.tex", "windows.tex", "constants.tex"): return R003 / "tables" / name, "R003 analysis/make_tables.py"
        return R002 / "tables" / name, "R002 tables/make_tables.py"
    if rel.startswith("external_data/"): return R002 / "analysis" / "results" / name, "R002 analysis/01_data.jl, 02_concept.jl, 03_landscape.jl, 04_sweeps.jl, 08_post.jl"
    return None
def inherited_files():
    out = []
    for d in ("figures", "external_data"):
        for p in sorted((RPT / d).rglob("*")):
            if p.is_file() and not p.name.startswith("."): out.append(str(p.relative_to(RPT)))
    return out
