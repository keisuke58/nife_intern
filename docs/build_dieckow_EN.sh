#!/usr/bin/env bash
# build_dieckow_EN.sh — rebuild ONLY dieckow_slides_EN (beamer PDF) with the
# same EN font setup as build_docs.sh. Compile from ROOT so figures/ paths resolve.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"      # .../nife/docs
ROOT="$(cd "$HERE/.." && pwd)"             # .../nife
export PATH="$HOME/texlive/2025/bin/x86_64-linux:/usr/bin:/bin"
export TEXMFHOME=/tmp/empty_texmf
mkdir -p "$TEXMFHOME"

STIX=/home/nishioka/texlive/2025/texmf-dist/fonts/opentype/public/stix2-otf
EN_FONTS=(-V mainfont="STIXTwoText-Regular.otf"
          -V mainfontoptions="Path=$STIX/, BoldFont=STIXTwoText-Bold.otf, ItalicFont=STIXTwoText-Italic.otf, BoldItalicFont=STIXTwoText-BoldItalic.otf"
          -V monofont="DejaVu Sans Mono" -H "$HERE/_en_symbols.tex")

cd "$ROOT"
pandoc "docs/dieckow_slides_EN.md" -t beamer -s "${EN_FONTS[@]}" -o "docs/dieckow_slides_EN.tex"
xelatex -interaction=nonstopmode -output-directory=docs "docs/dieckow_slides_EN.tex" >/tmp/dieckow_build.log 2>&1
xelatex -interaction=nonstopmode -output-directory=docs "docs/dieckow_slides_EN.tex" >/tmp/dieckow_build.log 2>&1
echo "done: $(ls -la "$HERE/dieckow_slides_EN.pdf")"
