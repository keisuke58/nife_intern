#!/usr/bin/env bash
# build_docs.sh — render the FISH-pipeline report + slides (JA & EN) to PDF, and
# slides to self-contained HTML.
#   JA: Japanese Mincho (Noto Serif CJK JP) body, Noto Sans Mono CJK JP code.
#   EN: clean Latin (DejaVu Serif / DejaVu Sans Mono), no CJK.
#
# NOTE on this machine's TeX: the personal TeX Live 2025 lives at ~/texlive/2025,
# but ~/texmf holds a STALE latex.ltx + l3kernel that shadow it and break modern
# packages (expl3 "extra }" / undefined \IfFormatAtLeastT). We bypass it with an
# empty TEXMFHOME and use the personal engine. (One-time: fmtutil-sys --byfmt
# xelatex with that TEXMFHOME, to rebuild the format on the clean kernel.)
# Image paths: reports use ../figures (compile from docs/), slides use figures/
# (compile from repo root) — hence the two working dirs.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"      # .../nife/docs
ROOT="$(cd "$HERE/.." && pwd)"             # .../nife

export PATH="$HOME/texlive/2025/bin/x86_64-linux:/usr/bin:/bin"
export TEXMFHOME=/tmp/empty_texmf
mkdir -p "$TEXMFHOME"

JA_FONTS=(-V mainfont="Noto Serif CJK JP" -V monofont="Noto Sans Mono CJK JP"
          -V CJKmainfont="Noto Serif CJK JP" -V CJKmonofont="Noto Sans Mono CJK JP")
# EN: Times-family text via STIX Two Text (loaded by explicit OTF path — it is
# not in fontconfig) + DejaVu mono. STIX Two *Text* lacks a few set/relation
# glyphs (∩ ≠ ×), so the symbol header routes those through math mode.
STIX=/home/nishioka/texlive/2025/texmf-dist/fonts/opentype/public/stix2-otf
EN_FONTS=(-V mainfont="STIXTwoText-Regular.otf"
          -V mainfontoptions="Path=$STIX/, BoldFont=STIXTwoText-Bold.otf, ItalicFont=STIXTwoText-Italic.otf, BoldItalicFont=STIXTwoText-BoldItalic.otf"
          -V monofont="DejaVu Sans Mono" -H "$HERE/_en_symbols.tex")

# Remove ONLY the per-document generated files (NOT the hand-written _en_symbols.tex).
clean() {
  for ext in tex aux log nav out snm toc vrb; do
    rm -f "$HERE"/fish_pipeline_*."$ext" 2>/dev/null || true
  done
}

build_slides() {  # $1=md stem, rest=font args ; beamer PDF + dzslides HTML
  local stem="$1"; shift
  cd "$ROOT"
  pandoc "docs/$stem.md" -t beamer -s "$@" -o "docs/$stem.tex"
  xelatex -interaction=nonstopmode -output-directory=docs "docs/$stem.tex" >/dev/null
  xelatex -interaction=nonstopmode -output-directory=docs "docs/$stem.tex" >/dev/null
  pandoc "docs/$stem.md" -t dzslides -s --self-contained --slide-level=2 \
    -o "docs/$stem.html"
}

build_report() {  # $1=md stem, rest=font args ; article PDF
  local stem="$1"; shift
  cd "$HERE"
  pandoc "$stem.md" -o "$stem.pdf" --pdf-engine=xelatex \
    -V geometry:margin=2.2cm -V linkcolor=blue "$@"
}

echo "[JA] slides + report (Mincho)"
build_slides fish_pipeline_slides       "${JA_FONTS[@]}"
build_report fish_pipeline_report       "${JA_FONTS[@]}"
echo "[EN] slides + report (DejaVu)"
build_slides fish_pipeline_slides_EN    "${EN_FONTS[@]}"
build_report fish_pipeline_report_EN    "${EN_FONTS[@]}"
clean

echo "done:"
ls -la "$HERE"/fish_pipeline_*.pdf "$HERE"/fish_pipeline_*.html
