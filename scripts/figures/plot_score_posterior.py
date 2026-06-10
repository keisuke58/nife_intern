#!/usr/bin/env python3
"""
plot_score_posterior.py — visualise ② score-based Langevin posterior for A.

Reads:   results/score_posterior/A_samples.npy  (n_samples, 100)
         results/score_posterior/posterior_summary.json
         results/dieckow_cr/fit_glv_hamilton_kegg_expanded_agora_w0p5.json (warm-start A)

Panels:
  A — posterior mean A matrix heatmap  vs  warm-start A heatmap
  B — per-element marginal histograms for selected (i,j) pairs
  C — sign agreement: Langevin posterior mean vs AGORA sign prior

Output: results/score_posterior/fig_score_posterior.pdf/png

Run (repo root, TeX Live on PATH):
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
    python scripts/figures/plot_score_posterior.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from thesis_style import use
from guild_replicator_dieckow import GUILD_SHORT_LIST, N_G

_here = _pathlib.Path(__file__).resolve().parents[2]
SAMPLES_NPY = _here / 'results' / 'score_posterior' / 'A_samples.npy'
SUMMARY_JSON = _here / 'results' / 'score_posterior' / 'posterior_summary.json'
WARMSTART_JSON = (_here / 'results' / 'dieckow_cr' /
                  'fit_glv_hamilton_kegg_expanded_agora_w0p5.json')
OUT_DIR = _here / 'results' / 'score_posterior'


def _plot_A_heatmap(ax, A, title, vmax=None):
    vm = vmax or np.abs(A).max()
    im = ax.imshow(A, cmap='RdBu_r', vmin=-vm, vmax=vm, aspect='auto')
    ax.set_xticks(range(N_G)); ax.set_yticks(range(N_G))
    ax.set_xticklabels(GUILD_SHORT_LIST, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(GUILD_SHORT_LIST, fontsize=6)
    ax.set_title(title, fontsize=8)
    return im


def main():
    if not SAMPLES_NPY.exists():
        print(f'No samples yet: {SAMPLES_NPY}')
        print('Run: python scripts/analysis/score_posterior_glv.py first')
        return

    samples = np.load(SAMPLES_NPY)           # (n_samples, 100)
    summary = json.load(open(SUMMARY_JSON))
    burnin  = len(samples) // 2
    post    = samples[burnin:]               # posterior half

    A_mean = np.array(summary['A_mean']).reshape(N_G, N_G)
    A_std  = np.array(summary['A_std']).reshape(N_G, N_G)

    # Load warm-start A for comparison
    A_warm = None
    if WARMSTART_JSON.exists():
        d = json.load(open(WARMSTART_JSON))
        A_warm = np.array(d['A'])

    fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.42))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.45)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])

    vmax = max(np.abs(A_mean).max(),
               np.abs(A_warm).max() if A_warm is not None else 0)

    im = _plot_A_heatmap(ax0, A_mean, 'Langevin posterior mean', vmax=vmax)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    if A_warm is not None:
        _plot_A_heatmap(ax1, A_warm, 'Warm-start A (L-BFGS-B)', vmax=vmax)
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    else:
        _plot_A_heatmap(ax1, A_std, 'Posterior std(A)', vmax=A_std.max())

    # Panel C: sign agreement per element
    sign_agree = np.sign(A_mean) == np.sign(A_warm) if A_warm is not None else None
    if sign_agree is not None:
        frac = sign_agree[~np.eye(N_G, dtype=bool)].mean()
        ax2.imshow(sign_agree.astype(float), cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        ax2.set_xticks(range(N_G)); ax2.set_yticks(range(N_G))
        ax2.set_xticklabels(GUILD_SHORT_LIST, rotation=45, ha='right', fontsize=6)
        ax2.set_yticklabels(GUILD_SHORT_LIST, fontsize=6)
        ax2.set_title(f'Sign agree vs warm-start\n(off-diag={frac:.0%})', fontsize=8)
    else:
        ax2.text(0.5, 0.5, 'no warm-start\nfor comparison',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=8)
        ax2.set_title('Sign agreement', fontsize=8)

    fig.suptitle(
        f'Score-based Langevin posterior  '
        f'(n={len(post)} post-burnin samples, '
        f'{summary["n_chains"]} chains, {summary["n_steps"]} steps)',
        fontsize=8, y=1.01,
    )
    fig.tight_layout()

    pdf = OUT_DIR / 'fig_score_posterior.pdf'
    png = OUT_DIR / 'fig_score_posterior.png'
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(png, dpi=200, bbox_inches='tight')
    print(f'Saved: {pdf}')
    print(f'Saved: {png}')
    plt.close(fig)


if __name__ == '__main__':
    main()
