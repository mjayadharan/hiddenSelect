#!/usr/bin/env python
"""make_report_md.py -- derive report.md (full-content Markdown rendition) from report.tex (authoritative).
Inlines the macro files and tables (Markdown twins where they exist), converts figure/table environments to
Markdown blocks with their full captions, runs pandoc latex->gfm, keeps math as LaTeX inside $...$.
"""
import re, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
tex = (ROOT / "report.tex").read_text()
# macros: inline both macro files so pandoc can expand them
for f in ("constants.tex", "numbers.tex"):
    tex = tex.replace(f"\\input{{tables/{f}}}", (ROOT / "tables" / f).read_text())
tex = tex.replace("\\input{tables/gates_status.tex}", (ROOT / "tables" / "gates_status.tex").read_text().strip())
def inline_table(m):
    name = m.group(1); md = ROOT / "tables" / f"{name}.md"
    return f"\n\nTABLEBLOCK{name}TABLEBLOCK\n\n" if md.exists() else (ROOT / "tables" / f"{name}.tex").read_text()
tex = re.sub(r"\\resizebox\{[^}]*\}\{!\}\{\\input\{tables/([^}]+)\.tex\}\}", lambda m: inline_table(m), tex)
tex = re.sub(r"\\input\{tables/([^}]+)\.tex\}", inline_table, tex)
figc, tabc = [0], [0]
def fig(m):
    body = m.group(1); path = re.search(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", body).group(1)
    cap = re.search(r"\\caption\{(.*)\}\\label\{([^}]+)\}", body, re.S); figc[0] += 1
    return f"\n\nFIGBLOCK{figc[0]}FIGSEP{path}FIGSEP{cap.group(2)}FIGSEP{cap.group(1)}FIGBLOCK\n\n"
tex = re.sub(r"\\begin\{figure\}\[H\](.*?)\\end\{figure\}", fig, tex, flags=re.S)
def tab(m):
    body = m.group(1); cap = re.search(r"\\caption\{(.*?)\}\\label\{([^}]+)\}", body, re.S); tabc[0] += 1
    rest = body[cap.end():] if cap else body
    return f"\n\nTABBLOCK{tabc[0]}FIGSEP{cap.group(2)}FIGSEP{cap.group(1)}TABBLOCK\n\n" + rest
tex = re.sub(r"\\begin\{table\}\[H\](.*?)\\end\{table\}", tab, tex, flags=re.S)
tex = re.sub(r"\\centering|\\scriptsize|\\small|\\needspace\{[^}]*\}|\\tableofcontents", "", tex)
tex = re.sub(r"\\readthis\{", r"\\textbf{Read this:} {", tex); tex = re.sub(r"\\fig\{([^}]+)\}", r"`figures/\1`", tex)
# algorithmic block -> verbatim-ish
tex = re.sub(r"\\begin\{algorithm\}\[H\](.*?)\\end\{algorithm\}", lambda m: "\n\nALGBLOCK" + m.group(1) + "ALGBLOCK\n\n", tex, flags=re.S)
tmp = ROOT / "analysis" / "_report_tmp.tex"; tmp.write_text(tex)
md = subprocess.run(["pandoc", "-f", "latex", "-t", "gfm+tex_math_dollars", "--wrap=none", str(tmp)], capture_output=True, text=True, check=True).stdout
tmp.unlink()
def figmd(m):
    n, path, label, caption = m.groups()
    return f"**Figure {n}** (`{path}`)\n\n![{label}]({path})\n\n*{caption.strip()}*\n"
md = re.sub(r"FIGBLOCK(\d+)FIGSEP(.*?)FIGSEP(.*?)FIGSEP(.*?)FIGBLOCK", figmd, md, flags=re.S)
md = re.sub(r"TABBLOCK(\d+)FIGSEP(.*?)FIGSEP(.*?)TABBLOCK", lambda m: f"**Table {m.group(1)}.** {m.group(3).strip()}\n", md, flags=re.S)
md = re.sub(r"TABLEBLOCK(.*?)TABLEBLOCK", lambda m: (ROOT / "tables" / f"{m.group(1)}.md").read_text().strip(), md)
def alg(m):
    body = m.group(1)
    body = re.sub(r"\\caption\{([^}]*)\}\\label\{[^}]*\}", r"\1", body); body = re.sub(r"\\begin\{algorithmic\}\[1\]|\\end\{algorithmic\}", "", body)
    body = body.replace("\\State", "").replace("\\For{", "for ").replace("\\EndFor", "end for").replace("\\Return", "return").replace("\\gets", "←")
    return "```\n" + "\n".join(l.strip() for l in body.splitlines() if l.strip()) + "\n```\n"
md = re.sub(r"ALGBLOCK(.*?)ALGBLOCK", alg, md, flags=re.S)
# cross references: pandoc leaves [label]; map to names
labels = {}
for kind, pat in (("Figure", r"FIGBLOCK"),):
    pass
figlabels = re.findall(r"\\caption\{.*?\}\\label\{(fig:[^}]+)\}", (ROOT / "report.tex").read_text(), re.S)
tablabels = re.findall(r"\\caption\{.*?\}\\label\{(tab:[^}]+)\}", (ROOT / "report.tex").read_text(), re.S)
for i, l in enumerate(figlabels, 1): md = md.replace(f"[{l}]", f"{i}")
for i, l in enumerate(tablabels, 1): md = md.replace(f"[{l}]", f"{i}")
secs = re.findall(r"\\(?:sub)?section\{([^}]+)\}\\label\{(sec:[^}]+)\}", (ROOT / "report.tex").read_text())
for i, (name, l) in enumerate(secs): md = md.replace(f"[{l}]", f"“{name}”")
thm = {"lem:trunc": "Lemma 1", "lem:vi": "Lemma 2", "lem:vi_local": "Lemma 3", "lem:rhobar": "Lemma 4",
       "lem:state": "Lemma 5", "lem:param": "Lemma 6", "lem:noescape": "Lemma 7", "lem:residual": "Lemma 8",
       "def:Tx": "Definition 1",
       "prop:cost": "Proposition 1", "prop:perr": "Proposition 2", "prop:removal": "Proposition 3",
       "thm:main": "Theorem 1",
       "cor:confined": "Corollary 1", "cor:removal_rho": "Corollary 2", "cor:removal_node": "Corollary 3",
       "cor:removal_global": "Corollary 4", "cor:basin": "Corollary 5", "cor:repartition": "Corollary 6",
       "cor:window": "Corollary 7",
       "rem:reconnect": "Remark 1", "rem:rhoplus": "Remark 2", "rem:impl": "Remark 3",
       "rem:repart_numbers": "Remark 4", "rem:envelopes": "Remark 5", "rem:weighted": "Remark 6",
       "alg:gp": "Algorithm 1"}
for l, n in thm.items(): md = md.replace(f"[{l}]", n)
md = re.sub(r"\[eq:([a-zA-Z0-9_]+)\]", r"(eq. \1)", md)
md = re.sub(r"<span style=\"color: [^\"]*\">(.*?)</span>", r"*\1*", md, flags=re.S)
header = ("# Sparse Optimization using Multiple Shooting and Guess Propagation — revised manuscript (R004)\n\n"
          "**Report ID:** R004 (`reports/2026-09-11-R004-revised-manuscript/`)  \n**Date:** 11 September 2026  \n"
          "**Authoritative source:** `report.tex` / `report.pdf` — this `.md` is a derived, full-content view generated by `analysis/make_report_md.py`; on disagreement the `.tex` wins.  \n"
          f"**Gates:** {(ROOT / 'tables' / 'gates_status.tex').read_text().strip()} — `analysis/results/gates_summary.json`\n\n"
          "Theorem-like statements, matching the PDF: Definition 1 (confinement time); Lemma 1 (conditioning on $E_X$), Lemma 2 (minimiser inequality), Lemma 3 (local minimiser inequality), Lemma 4 (envelope properties), Lemma 5 (flow sensitivity to the state, local), Lemma 6 (flow sensitivity to the parameters, local), Lemma 7 (no escape from the tube), Lemma 8 (largest residual of a window); Proposition 1 (cost at the truth), Proposition 2 (minimiser vs truth), Proposition 3 (node removal, per datum); Theorem 1 (minimiser under node removal); Corollary 1 (full-window estimates), Corollary 2 (uniform-noise form), Corollary 3 (per-node form), Corollary 4 (collapsed form), Corollary 5 (basin retention), Corollary 6 (re-partition), Corollary 7 (per-window exponents); Remark 1 (reconnection), Remark 2 (smooth majorant), Remark 3 (implemented objective), Remark 4 (re-partition numbers), Remark 5 (envelopes), Remark 6 (weighted norms).\n\n")
(ROOT / "report.md").write_text(header + md)
print("report.md:", len(md), "chars;", figc[0], "figures;", tabc[0], "tables")
