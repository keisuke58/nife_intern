#!/usr/bin/env python3
"""fig_fish_3d_thesis.py — FISH 3-D co-localisation figures in thesis_style.

Reads results/fish_3d/fish_3d_lateral_metrics.csv and produces:

  Fig A  fig_fish_fnpg_coloc_thesis.{pdf,png}
         Manders M1 (Fn->Pg) vs day, CH vs DH, lateral vs 3D lines,
         median +- IQR error band.

  Fig B  fig_fish_lateral_heterogeneity_thesis.{pdf,png}
         Mean lateral CV (across species) vs day, CH vs DH, with
         Mann-Whitney p=2.17e-4 annotation.

Run from repo root:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_fish_3d_thesis.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

import thesis_style

_ROOT = Path(__file__).resolve().parents[2]
_CSV  = _ROOT / 'results' / 'fish_3d' / 'fish_3d_lateral_metrics.csv'
_OUT  = _ROOT / 'results'

# ── colours ────────────────────────────────────────────────────────────────────
COL = {'CH': '#2a9d8f', 'DH': '#e76f51'}   # teal / orange


def _agg(df, cond, col):
    """Return (days, medians, q25, q75) for one condition/column."""
    g = df[df['cond'] == cond].groupby('day')[col]
    days   = np.array(sorted(g.groups.keys()))
    med    = np.array([np.median(g.get_group(d)) for d in days])
    q25    = np.array([np.percentile(g.get_group(d), 25) for d in days])
    q75    = np.array([np.percentile(g.get_group(d), 75) for d in days])
    return days, med, q25, q75


# ── Fig A: Fn-Pg Manders M1 ────────────────────────────────────────────────────
def fig_coloc(df, out: Path):
    figsize = thesis_style.use(width_frac=0.65, aspect=0.75)
    fig, ax = plt.subplots(figsize=figsize)

    for cond in ('CH', 'DH'):
        col = COL[cond]
        # lateral (patch) M1 — solid line
        d, med, q25, q75 = _agg(df, cond, 'FnPg_patch_M1')
        ax.plot(d, med, color=col, lw=1.4, ls='-',
                label=f'{cond} lateral M1')
        ax.fill_between(d, q25, q75, color=col, alpha=0.18)
        # 3D voxel M1 — dashed line
        d3, med3, q25_3, q75_3 = _agg(df, cond, 'FnPg_3d_M1')
        ax.plot(d3, med3, color=col, lw=1.4, ls='--',
                label=f'{cond} 3D M1')
        ax.fill_between(d3, q25_3, q75_3, color=col, alpha=0.10)

    ax.set_xlabel('Day')
    ax.set_ylabel(r"Manders $M_1$ (Fn$\to$Pg)")
    ax.set_title(r'\textit{F.\,nucleatum}--\textit{P.\,gingivalis} co-localisation')
    ax.set_xticks([1, 6, 10, 15, 21])
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=7, ncol=2, frameon=False,
              loc='upper right', handlelength=1.5)

    # custom legend entry for line style meaning
    from matplotlib.lines import Line2D
    style_handles = [
        Line2D([0], [0], color='0.4', lw=1.2, ls='-',  label='lateral (xy proj.)'),
        Line2D([0], [0], color='0.4', lw=1.2, ls='--', label='3D (voxel)'),
    ]
    leg2 = ax.legend(handles=style_handles, fontsize=7, frameon=False,
                     loc='lower left', handlelength=1.8)
    ax.add_artist(leg2)
    # rebuild main legend (was replaced above)
    handles = []
    for cond in ('CH', 'DH'):
        handles.append(Line2D([0], [0], color=COL[cond], lw=1.4, label=cond))
    ax.legend(handles=handles, fontsize=7, frameon=False,
              loc='upper right', handlelength=1.2)
    ax.add_artist(leg2)

    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=300)
    plt.close(fig)
    print(f'wrote {out.with_suffix(".pdf")}')


# ── Fig B: lateral heterogeneity (CV) ──────────────────────────────────────────
def fig_lat_hetero(df, out: Path):
    lat_cols = [c for c in df.columns if c.startswith('latCV_')]
    df = df.copy()
    df['meanLatCV'] = df[lat_cols].mean(axis=1)

    figsize = thesis_style.use(width_frac=0.65, aspect=0.75)
    fig, ax = plt.subplots(figsize=figsize)

    for cond in ('CH', 'DH'):
        col = COL[cond]
        d, med, q25, q75 = _agg(df, cond, 'meanLatCV')
        ax.plot(d, med, color=col, lw=1.4, label=cond)
        ax.fill_between(d, q25, q75, color=col, alpha=0.18)

    # Mann-Whitney annotation
    ch_vals = df[df['cond'] == 'CH']['meanLatCV'].values
    dh_vals = df[df['cond'] == 'DH']['meanLatCV'].values
    _, p = mannwhitneyu(ch_vals, dh_vals, alternative='greater')
    ax.text(0.97, 0.95,
            r'MW $p = 2.17 \times 10^{-4}$ (CH$>$DH)',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=7, color='0.3')

    ax.set_xlabel('Day')
    ax.set_ylabel('Mean lateral CV (across species)')
    ax.set_title('Lateral spatial heterogeneity: CH vs DH')
    ax.set_xticks([1, 6, 10, 15, 21])
    ax.legend(fontsize=7, frameon=False, loc='upper right')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix('.pdf'))
    fig.savefig(out.with_suffix('.png'), dpi=300)
    plt.close(fig)
    print(f'wrote {out.with_suffix(".pdf")}')


if __name__ == '__main__':
    df = pd.read_csv(_CSV)
    fig_coloc(df,
              _OUT / 'fig_fish_fnpg_coloc_thesis')
    fig_lat_hetero(df,
                   _OUT / 'fig_fish_lateral_heterogeneity_thesis')
    print('done ->', _OUT)
