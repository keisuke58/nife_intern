# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""loo_stability_analysis.py — A-matrix sign stability across 10 LOO folds (1-B).

For each off-diagonal pair (i,j) in the 10-guild interaction matrix:
  - sign consensus: how many of 10 folds agree with majority sign
  - mean A_ij and std across folds
  - regime: aligned / prior-constrained / data-driven / uncertain

Output:
  results/fig_loo_stability.{pdf,png}
  results/loo_stability_stats.json

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/analysis/loo_stability_analysis.py
"""

import json, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

from thesis_style import use, clean_ax
from guild_replicator_dieckow import GUILD_ORDER

ROOT     = pathlib.Path(__file__).parents[2]
LOODIR   = ROOT / "results" / "dieckow_cr"
OUT      = ROOT / "results" / "fig_loo_stability"
OUT_JSON = ROOT / "results" / "loo_stability_stats.json"

PREFIX = "loo_expanded_agora_w1p0"
N_G    = 10

SHORT = ['Actin.', 'Bacil.', 'Bact.', 'β-Prot.',
         'Clost.', 'Corio.', 'Fusob.', 'γ-Prot.', 'Negat.', 'Other']


def load_folds():
    files = sorted(glob.glob(str(LOODIR / f"{PREFIX}_fold*.json")))
    assert len(files) == 10, f"Expected 10 folds, got {len(files)}"
    return np.stack([np.array(json.load(open(f))['A']) for f in files], axis=0)


def load_prior():
    d = json.load(open(LOODIR / "fit_glv_hamilton_kegg_expanded_agora_w1p0.json"))
    return np.array(d['net_flow'])


def main():
    A_folds = load_folds()           # (10, 10, 10): folds × guilds × guilds
    F       = load_prior()           # (10, 10)
    A_mean  = A_folds.mean(axis=0)
    A_std   = A_folds.std(axis=0)

    # sign consensus per pair: 5–10 (how many folds match majority sign)
    A_signs   = np.sign(A_folds)
    consensus = np.zeros((N_G, N_G))
    for i in range(N_G):
        for j in range(N_G):
            if i == j:
                continue
            s = A_signs[:, i, j]
            consensus[i, j] = max((s > 0).sum(), (s < 0).sum())

    # regime classification
    regime = np.full((N_G, N_G), '', dtype=object)
    for i in range(N_G):
        for j in range(N_G):
            if i == j:
                continue
            constrained = F[i, j] != 0
            stable      = consensus[i, j] >= 8
            if constrained and stable:
                regime[i, j] = 'aligned'
            elif constrained:
                regime[i, j] = 'prior-constrained'
            elif stable:
                regime[i, j] = 'data-driven'
            else:
                regime[i, j] = 'uncertain'

    # ── figure ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=use(width_frac=1.0, aspect=0.44),
                             gridspec_kw={'width_ratios': [1.05, 1]})

    diag_mask = np.eye(N_G, dtype=bool)

    # Panel A: sign consensus (5–10)
    ax = axes[0]
    cmap_cons = LinearSegmentedColormap.from_list(
        'cons', ['#f7f7f7', '#92c5de', '#0571b0'], N=6)
    ax.imshow(np.where(diag_mask, np.nan, consensus),
              vmin=4.5, vmax=10, cmap=cmap_cons, aspect='equal')

    for i in range(N_G):
        for j in range(N_G):
            if i == j:
                ax.add_patch(mpatches.Rectangle(
                    (j-.5, i-.5), 1, 1, color='0.88', zorder=0))
                continue
            c = int(consensus[i, j])
            ax.text(j, i, str(c), ha='center', va='center', fontsize=5.5,
                    color='white' if c >= 9 else '0.25',
                    fontweight='bold' if c >= 9 else 'normal')
            if F[i, j] != 0:
                ax.add_patch(mpatches.Rectangle(
                    (j-.48, i-.48), .96, .96,
                    fill=False, edgecolor='#e66101', linewidth=0.8))

    ax.set_xticks(range(N_G))
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=6)
    ax.set_yticks(range(N_G))
    ax.set_yticklabels(SHORT, fontsize=6)
    ax.set_title(r'\textbf{A}\ Sign consensus (of 10 folds)', fontsize=8, loc='left')

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    cb = fig.colorbar(ScalarMappable(norm=Normalize(4.5, 10), cmap=cmap_cons),
                      ax=ax, fraction=0.046, pad=0.02)
    cb.set_label('consensus', fontsize=7)
    cb.ax.tick_params(labelsize=6)
    ax.add_patch(mpatches.Rectangle((0,0),0,0, fill=False,
                 edgecolor='#e66101', lw=0.8, label='AGORA-constrained'))
    ax.legend(loc='lower right', fontsize=6, framealpha=0.9, edgecolor='0.8',
              bbox_to_anchor=(1.0, 0.0))

    # Panel B: mean A (diverging)
    ax2 = axes[1]
    vmax = np.nanmax(np.abs(np.where(diag_mask, np.nan, A_mean)))
    im2  = ax2.imshow(np.where(diag_mask, np.nan, A_mean),
                      vmin=-vmax, vmax=vmax, cmap='RdBu_r', aspect='equal')
    for i in range(N_G):
        for j in range(N_G):
            if i == j:
                ax2.add_patch(mpatches.Rectangle(
                    (j-.5, i-.5), 1, 1, color='0.88', zorder=0))
                continue
            ax2.text(j, i, f'{A_mean[i,j]:.2f}', ha='center', va='center',
                     fontsize=5.5,
                     color='white' if abs(A_mean[i,j]) > vmax*.5 else '0.2')
    ax2.set_xticks(range(N_G))
    ax2.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=6)
    ax2.set_yticks(range(N_G))
    ax2.set_yticklabels([], fontsize=6)
    ax2.set_title(r'\textbf{B}\ Mean $A_{ij}$ across folds', fontsize=8, loc='left')
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
    cb2.set_label('$A_{ij}$', fontsize=7)
    cb2.ax.tick_params(labelsize=6)

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}')
        print(f'saved → {OUT}.{ext}')

    # ── console summary ───────────────────────────────────────────────────────
    counts = {r: int((regime == r).sum())
              for r in ['aligned', 'prior-constrained', 'data-driven', 'uncertain']}
    print('\nRegime counts (off-diagonal pairs, n=90):')
    for r, n in counts.items():
        print(f'  {r:22s}: {n:3d}')

    print('\nData-driven stable pairs (consensus ≥ 8, unconstrained):')
    for i in range(N_G):
        for j in range(N_G):
            if regime[i, j] == 'data-driven':
                print(f'  {SHORT[i]:8s}→{SHORT[j]:8s}  '
                      f'A={A_mean[i,j]:+.3f}  cons={int(consensus[i,j])}')

    OUT_JSON.write_text(json.dumps({
        'regime_counts': counts,
        'consensus_matrix': consensus.tolist(),
        'A_mean': A_mean.tolist(),
        'A_std':  A_std.tolist(),
    }, indent=2))
    print(f'\nstats → {OUT_JSON}')


if __name__ == '__main__':
    main()
