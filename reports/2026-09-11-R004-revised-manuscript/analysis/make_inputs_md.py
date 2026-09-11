"""make_inputs_md.py -- write INPUTS.md: every inherited file with its SHA-256, source path and producer."""
import hashlib, subprocess
from sources import RPT, REPORTS, inherited_files, source_of
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=RPT, capture_output=True, text=True).stdout.strip()
rows = []
for rel in inherited_files():
    src, note = source_of(rel)
    rows.append(f"| `{rel}` | `{sha(RPT / rel)}` | `{src.relative_to(REPORTS)}` | {note} |")
(RPT / "INPUTS.md").write_text(f"""# R004 — inputs

R004 computes nothing new. Every figure, table and number is inherited from R002 and R003 (both in
`reports/`, repo HEAD `{head}` at the time of writing) and copied **byte for byte** into this folder;
gates G1, G2 and G4 assert the byte identity. `deps/` is intentionally empty: the two Python scripts
import only numpy/pandas and read no Julia source. No file is larger than 1 MB.

| file in this folder | sha256 | source (relative to `reports/`) | produced by |
|---|---|---|---|
""" + "\n".join(rows) + "\n")
print(len(rows), "files listed")
