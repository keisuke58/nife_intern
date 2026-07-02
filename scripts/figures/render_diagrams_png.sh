#!/usr/bin/env bash
# Rasterise the diagram SVGs (docs/diagrams/*.svg) to PNG with headless
# Chromium. No LaTeX/ImageMagick needed; 2x device scale for crisp output.
#
#   scripts/figures/render_diagrams_png.sh [OUTDIR]
#
# CHROME env var overrides the browser path.
set -euo pipefail

OUTDIR="${1:-docs/diagrams}"
CHROME="${CHROME:-/opt/pw-browsers/chromium-1194/chrome-linux/chrome}"
[ -x "$CHROME" ] || CHROME="$(command -v chromium || command -v chromium-browser || command -v google-chrome)"

# viewport sizes must match the SVG width/height declared in the generator
declare -A SIZE=(
  [fig1_pillars]="900,560"
  [fig2_pipeline]="820,600"
  [fig3_sign_prior]="960,400"
  [fig4_attractors]="720,600"
)

for name in "${!SIZE[@]}"; do
  svg="$OUTDIR/$name.svg"
  png="$OUTDIR/$name.png"
  [ -f "$svg" ] || { echo "missing $svg" >&2; exit 1; }
  "$CHROME" --headless --disable-gpu --no-sandbox --hide-scrollbars \
    --force-device-scale-factor=2 --default-background-color=FFFFFFFF \
    --window-size="${SIZE[$name]}" \
    --screenshot="$png" "file://$(realpath "$svg")" >/dev/null 2>&1
  echo "rendered $png"
done
