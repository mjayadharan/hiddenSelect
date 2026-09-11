#!/usr/bin/env bash
# Convert a Markdown report to a PDF (the standard .md + .pdf deliverable pair).
#
#   Usage: scripts/md2pdf.sh <input.md> [output.pdf]
#
# Requires: pandoc + a TeX engine (xelatex). On macOS:
#   brew install pandoc      # xelatex comes with MacTeX
#
# Uses STIX Two fonts (ship with MacTeX) for full Unicode/math coverage, plus
# scripts/pdf-header.tex for a few arrow glyphs text fonts omit.
set -euo pipefail

SRC="${1:?usage: md2pdf.sh <input.md> [output.pdf]}"
OUT="${2:-${SRC%.md}.pdf}"
HDR="$(cd "$(dirname "$0")" && pwd)/pdf-header.tex"

pandoc "$SRC" -o "$OUT" \
  --pdf-engine=xelatex \
  -V geometry:margin=1in -V fontsize=11pt \
  -V colorlinks=true -V linkcolor=blue -V urlcolor=blue \
  -V mainfont="STIX Two Text" -V mathfont="STIX Two Math" \
  -V monofont="Menlo" \
  -H "$HDR"

echo "wrote $OUT"
