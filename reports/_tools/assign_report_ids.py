#!/usr/bin/env python
"""assign_report_ids.py — assign and maintain unique report IDs (R###).

STANDARD (owner mandate 2026-07-23): every report folder under reports/
carries a unique ID embedded in its folder name directly after the date,

    <YYYY-MM-DD>-R###-<topic>

plus a REPORT_ID.md file inside the folder, and a row in
reports/REPORT_INDEX.{md,json}. IDs are assigned once, in chronological
order (date prefix, then first-commit time, then name), and are NEVER
reused or renumbered. New reports: create the folder WITH the next free
ID already in its name (see REPORT_INDEX.md for the highest ID), or
create it un-ID'd and run this tool once.

Running with no arguments is idempotent:
  1. finds report folders lacking an ID and assigns the next free ones;
  2. renames them (git mv when tracked, plain rename otherwise);
  3. rewrites repo-internal references to each old folder name in every
     TEXT file in the repo (binary files and .git are skipped; a
     boundary guard prevents prefix collisions such as
     ...-dominant-balance vs ...-dominant-balance-lgrid-lambda);
  4. writes REPORT_ID.md into each newly ID'd folder;
  5. regenerates REPORT_INDEX.md and REPORT_INDEX.json.

`--check` verifies (no un-ID'd folders, no duplicate IDs, index in sync)
without writing.
"""
from __future__ import annotations
import json
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
RPTS = TOOLS.parent
REPO = RPTS.parent

DATED = re.compile(r"^(\d{4}-\d{2}-\d{2})-(.+)$")
WITH_ID = re.compile(r"^(\d{4}-\d{2}-\d{2})-(R\d{3})-(.+)$")
BINARY_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".mp4", ".npz",
              ".npy", ".zip", ".gz", ".pyc", ".ipynb_checkpoints",
              ".parquet", ".pkl", ".synctex"}
MAX_REWRITE_BYTES = 8_000_000


def report_dirs():
    return sorted(d for d in RPTS.iterdir()
                  if d.is_dir() and DATED.match(d.name))


def first_commit_ts(path: Path) -> int:
    out = subprocess.run(
        ["git", "log", "--format=%at", "--reverse", "--", str(path)],
        cwd=REPO, capture_output=True, text=True).stdout.strip()
    return int(out.splitlines()[0]) if out else 9_999_999_999


def dir_tracked(path: Path) -> bool:
    """True if git tracks at least one file under path."""
    out = subprocess.run(["git", "ls-files", str(path)], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    return bool(out)


def text_files():
    for p in REPO.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(REPO).as_posix()
        if rel.startswith(".git/"):
            continue
        if p.suffix.lower() in BINARY_EXT:
            continue
        if p.stat().st_size > MAX_REWRITE_BYTES:
            continue
        try:
            head = p.open("rb").read(8192)
        except OSError:
            continue
        if b"\0" in head:
            continue
        yield p


def rewrite_references(mapping: dict[str, str]):
    """Replace old folder names with new ones in every repo text file.
    Longest-first + a boundary guard ((?![A-Za-z0-9-])) so a name that is
    a prefix of another folder's name never clobbers it."""
    pats = [(re.compile(re.escape(old) + r"(?![A-Za-z0-9-])"), new)
            for old, new in
            sorted(mapping.items(), key=lambda kv: -len(kv[0]))]
    touched = []
    for p in text_files():
        try:
            txt = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_txt = txt
        for pat, new in pats:
            new_txt = pat.sub(new, new_txt)
        if new_txt != txt:
            p.write_text(new_txt, encoding="utf-8")
            touched.append(p.relative_to(REPO).as_posix())
    return touched


def write_report_id_md(folder: Path, rid: str, dt: str, slug: str,
                       former: str):
    (folder / "REPORT_ID.md").write_text(
        f"# Report ID: {rid}\n\n"
        f"- id: {rid}\n"
        f"- date: {dt}\n"
        f"- slug: {slug}\n"
        f"- folder: {folder.name}\n"
        f"- former_folder: {former}\n"
        f"- assigned: {date.today().isoformat()} by "
        f"reports/_tools/assign_report_ids.py (IDs are permanent; see "
        f"reports/REPORT_INDEX.md)\n")


def regen_index():
    rows = []
    for d in report_dirs():
        m = WITH_ID.match(d.name)
        if not m:
            continue
        dt, rid, slug = m.groups()
        former = ""
        idf = d / "REPORT_ID.md"
        if idf.exists():
            fm = re.search(r"former_folder: (.+)", idf.read_text())
            former = fm.group(1).strip() if fm else ""
        rows.append({"id": rid, "date": dt, "slug": slug,
                     "folder": d.name, "former_folder": former})
    rows.sort(key=lambda r: r["id"])
    (RPTS / "REPORT_INDEX.json").write_text(
        json.dumps(rows, indent=1) + "\n")
    md = ["# Report index — unique IDs",
          "",
          "One row per report. IDs are permanent (assigned "
          "chronologically, never reused or renumbered); use them to "
          "compile results across reports. New reports take the next "
          "free ID in the folder name (`<date>-R###-<topic>`) — see "
          "`reports/_tools/assign_report_ids.py`.",
          "",
          "| id | date | folder |",
          "|---|---|---|"]
    md += [f"| {r['id']} | {r['date']} | [{r['folder']}]({r['folder']}/) |"
           for r in rows]
    (RPTS / "REPORT_INDEX.md").write_text("\n".join(md) + "\n")
    return rows


def check():
    dirs = report_dirs()
    bad = [d.name for d in dirs if not WITH_ID.match(d.name)]
    ids = [WITH_ID.match(d.name).group(2) for d in dirs
           if WITH_ID.match(d.name)]
    dupes = {i for i in ids if ids.count(i) > 1}
    missing_md = [d.name for d in dirs
                  if WITH_ID.match(d.name)
                  and not (d / "REPORT_ID.md").exists()]
    idx = RPTS / "REPORT_INDEX.json"
    stale = (not idx.exists() or
             {r["folder"] for r in json.loads(idx.read_text())} !=
             {d.name for d in dirs if WITH_ID.match(d.name)})
    ok = not (bad or dupes or missing_md or stale)
    print(f"un-ID'd folders: {bad or 'none'}\nduplicate IDs: "
          f"{sorted(dupes) or 'none'}\nmissing REPORT_ID.md: "
          f"{missing_md or 'none'}\nindex stale: {stale}")
    return 0 if ok else 1


def main():
    if "--check" in sys.argv:
        sys.exit(check())
    dirs = report_dirs()
    used = {WITH_ID.match(d.name).group(2)
            for d in dirs if WITH_ID.match(d.name)}
    next_n = max([int(i[1:]) for i in used], default=0) + 1
    todo = [d for d in dirs if not WITH_ID.match(d.name)]
    todo.sort(key=lambda d: (DATED.match(d.name).group(1),
                             first_commit_ts(d), d.name))
    mapping, renames = {}, []
    for d in todo:
        dt, slug = DATED.match(d.name).groups()
        rid = f"R{next_n:03d}"
        next_n += 1
        new_name = f"{dt}-{rid}-{slug}"
        mapping[d.name] = new_name
        renames.append((d, RPTS / new_name, rid, dt, slug))
    for old, new, rid, dt, slug in renames:
        if dir_tracked(old):
            subprocess.run(["git", "mv", str(old), str(new)], cwd=REPO,
                           check=True)
        else:
            old.rename(new)
        print(f"{rid}  {old.name} -> {new.name}")
    if mapping:
        touched = rewrite_references(mapping)
        print(f"references rewritten in {len(touched)} files")
    for _, new, rid, dt, slug in renames:
        write_report_id_md(new, rid, dt, slug,
                           f"{dt}-{slug}")
    rows = regen_index()
    print(f"index regenerated: {len(rows)} reports")


if __name__ == "__main__":
    main()
