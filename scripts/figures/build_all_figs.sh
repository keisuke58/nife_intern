#!/usr/bin/env bash
# build_all_figs.sh — regenerate all thesis figures from scratch
# Run from repo root: bash scripts/figures/build_all_figs.sh
# Requires: texlive on PATH (set below), all data in results/

set -e
export PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

echo "=== Thesis figure build ==="
echo "Root: $ROOT"
echo ""

run() {
    echo "→ $1"
    python "$1"
}

# --- Main figures ---
run scripts/figures/generate_fig127.py          # Fig 1: study design; Fig 2: Dieckow data; Fig 7: 3D FISH
run scripts/figures/generate_fig3456.py         # Fig 3: LOO-RMSE; Fig 4: W sweep; Fig 5–6: network
run scripts/figures/generate_fig8.py            # Fig 8: PDE CH diffusion profile

# Run individually if generate_fig*.py are wrappers that may fail partially:
run scripts/figures/fig_loo_rmse_comparison.py  # Fig 3 (standalone)
run scripts/figures/fig_w_sweep_thesis.py       # Fig 4 (standalone)
run scripts/figures/fig_prior_asymmetry.py      # Fig 5
run scripts/figures/fig_network_thesis.py       # Fig 6
run scripts/figures/fig_fish_3d_thesis.py       # Fig 7 (standalone)
run scripts/figures/fig_pde_ch_profile.py       # Fig 8 (standalone)

# --- Supplementary ---
run scripts/analysis/loo_stability_analysis.py  # Supp A: LOO stability heatmap
run scripts/figures/fig_agora_slides.py         # Supp B: AGORA FBA network

echo ""
echo "=== Done. PDFs in results/ ==="
ls results/fig_*.pdf results/fig_loo_stability.pdf 2>/dev/null
