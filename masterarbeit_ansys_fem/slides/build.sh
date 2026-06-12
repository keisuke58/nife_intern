#!/usr/bin/env bash
# Build the FE-coupling slide deck (Beamer Madrid/whale, STIX via fontspec -> xelatex).
set -e
export PATH="$HOME/texlive/2025/bin/x86_64-linux:/usr/bin:/bin"
export TEXMFHOME=/tmp/empty_texmf; mkdir -p "$TEXMFHOME"
cd "$(dirname "$0")"
xelatex -interaction=nonstopmode slides_fem_coupling_en.tex
xelatex -interaction=nonstopmode slides_fem_coupling_en.tex
