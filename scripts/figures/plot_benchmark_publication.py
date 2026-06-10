#!/usr/bin/env python3
"""
plot_benchmark_publication.py — publication-grade benchmark figure.

Two panels (preserves the simpler plot_benchmark_baselines.py):
  (a) ranked horizontal bars: mean ± SD LOO-RMSE, coloured by model family,
      with a family legend and a dashed reference line at the best gLV.
  (b) paired per-fold comparison vs the best model (gLV+AGORA): each of the 10
      LOO folds (patients) is one connected pair; a one-sided Wilcoxon signed-
      rank p-value annotates how reliably gLV wins per patient.

Reads:  results/benchmark_baselines/rmse_table.json
Output: results/benchmark_baselines/fig_benchmark_publication.pdf/png

Run (repo root, TeX Live on PATH):
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
    python scripts/figures/plot_benchmark_publication.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.stats import wilcoxon
from thesis_style import use, PALETTE, clean_ax

_here   = _pathlib.Path(__file__).resolve().parents[2]
IN_JSON = _here / 'results' / 'benchmark_baselines' / 'rmse_table.json'
OUT_DIR = _here / 'results' / 'benchmark_baselines'

# family: (display label, colour) — thesis PALETTE for cross-figure consistency
FAMILY = {
    'mech': ('Mechanistic ODE', PALETTE['health']),     # teal  — the "good" model
    'ml':   ('ML regression',   PALETTE['dysbiosis']),  # orange
    'gen':  ('Generative',      '#6a3d9a'),             # purple
}

# key -> (display label, family)
METHODS = {
    'gLV + AGORA':                 ('gLV + AGORA',     'mech'),
    'gLV (no prior)':              ('gLV (no prior)',  'mech'),
    'gLV + L1/L2 prior':           ('gLV + L1/L2',     'mech'),
    'random_forest':               ('Random Forest',   'ml'),
    'mean_train':                  ('Train mean',      'ml'),
    'persistence':                 ('Persistence',     'ml'),
    'lasso':                       ('LASSO',           'ml'),
    'ridge':                       ('Ridge',           'ml'),
    'Flow Matching (conditional)': ('Flow Matching',   'gen'),
    'DDPM (conditional)':          ('DDPM',            'gen'),
}

REF_KEY = 'gLV + AGORA'   # best model, used as the paired baseline


def main():
    data = json.load(open(IN_JSON))

    # sort by mean RMSE ascending (best at top of barh)
    keys = sorted([k for k in METHODS if k in data],
                  key=lambda k: data[k]['mean_rmse'])

    means  = np.array([data[k]['mean_rmse'] for k in keys])
    stds   = np.array([data[k]['std_rmse']  for k in keys])
    labels = [METHODS[k][0] for k in keys]
    colors = [FAMILY[METHODS[k][1]][1] for k in keys]

    fig = plt.figure(figsize=use(width_frac=1.0, aspect=0.46))
    gs  = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.42)
    axa = fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1])

    # ── Panel (a): ranked horizontal bars ────────────────────────────────────
    y = np.arange(len(keys))[::-1]   # best at top
    axa.barh(y, means, xerr=stds, color=colors, height=0.66, zorder=3,
             error_kw=dict(ecolor='#333333', elinewidth=0.8, capsize=2),
             edgecolor='white', linewidth=0.5)
    for yi, m, s in zip(y, means, stds):
        axa.text(m + s + 0.004, yi, f'{m:.3f}', va='center', ha='left',
                 fontsize=6.2, color='#222222')

    ref = data[REF_KEY]['mean_rmse']
    axa.axvline(ref, color=FAMILY['mech'][1], lw=0.8, ls='--', zorder=2, alpha=0.7)

    axa.set_yticks(y)
    axa.set_yticklabels(labels, fontsize=7.5)
    axa.set_xlabel('LOO-CV RMSE  (mean $\\pm$ SD over 10 patients)', fontsize=8)
    axa.set_xlim(0, (means + stds).max() * 1.18)
    axa.xaxis.grid(True, lw=0.4, color='#cccccc', zorder=0)
    axa.set_axisbelow(True)
    clean_ax(axa)
    axa.tick_params(labelsize=7)

    # legend pinned to the top-right corner — the three short gLV bars sit at
    # the top, so the right half of those rows is guaranteed empty.
    legend_handles = [Patch(facecolor=c, label=lab) for lab, c in FAMILY.values()]
    axa.legend(handles=legend_handles, fontsize=6.3, loc='upper right',
               bbox_to_anchor=(1.0, 1.0), frameon=False,
               handlelength=1.0, handleheight=1.0, borderpad=0.2, labelspacing=0.3)
    axa.set_title('(a) Predictive accuracy ranking', fontsize=8, loc='left')

    # ── Panel (b): paired per-fold vs best gLV ───────────────────────────────
    ref_fold = np.array(data[REF_KEY]['per_fold'])
    # compare the best gLV against one representative of each other family
    comp_keys = ['random_forest', 'Flow Matching (conditional)', 'DDPM (conditional)']
    comp_keys = [k for k in comp_keys if k in data]

    xpos = np.arange(len(comp_keys) + 1)
    # column 0 = reference gLV, then each competitor
    all_cols = [REF_KEY] + comp_keys
    col_labels = ['gLV\n+AGORA'] + [METHODS[k][0].replace(' ', '\n') for k in comp_keys]
    col_colors = [FAMILY[METHODS[k][1]][1] for k in all_cols]

    # draw per-fold points + connecting lines from gLV to each competitor
    for fold in range(10):
        gv = ref_fold[fold]
        for ci, k in enumerate(comp_keys, start=1):
            cv = data[k]['per_fold'][fold]
            axb.plot([0, ci], [gv, cv], color='#bbbbbb', lw=0.4, zorder=1, alpha=0.7)

    for ci, k in enumerate(all_cols):
        vals = np.array(data[k]['per_fold'])
        jitter = (np.random.RandomState(ci).rand(10) - 0.5) * 0.12
        axb.scatter(np.full(10, ci) + jitter, vals, s=11,
                    color=col_colors[ci], zorder=3, edgecolor='white', linewidth=0.3)
        # mean marker
        axb.scatter([ci], [vals.mean()], marker='_', s=260,
                    color='#000000', zorder=4, linewidth=1.3)

    # Wilcoxon one-sided: gLV < competitor (gLV better) per patient
    ann_y = max(max(data[k]['per_fold']) for k in all_cols) * 1.04
    for ci, k in enumerate(comp_keys, start=1):
        cv = np.array(data[k]['per_fold'])
        try:
            stat, p = wilcoxon(ref_fold, cv, alternative='less')
        except ValueError:
            p = np.nan
        wins = int((ref_fold < cv).sum())
        star = ('***' if p < 1e-3 else '**' if p < 1e-2 else
                '*' if p < 0.05 else 'n.s.')
        axb.text(ci, ann_y, f'{wins}/10\n{star}', ha='center', va='bottom',
                 fontsize=6.0, color='#333333')

    axb.set_xticks(xpos)
    axb.set_xticklabels(col_labels, fontsize=6.6)
    axb.set_ylabel('LOO-CV RMSE (per patient)', fontsize=8)
    axb.set_ylim(0, ann_y * 1.16)
    axb.yaxis.grid(True, lw=0.4, color='#cccccc', zorder=0)
    axb.set_axisbelow(True)
    clean_ax(axb)
    axb.tick_params(labelsize=7)
    axb.set_title('(b) Paired per-patient comparison', fontsize=8, loc='left')
    axb.text(0.5, -0.30, 'wins/10 patients; Wilcoxon $p$: *** $<$0.001, ** $<$0.01, * $<$0.05',
             transform=axb.transAxes, ha='center', va='top', fontsize=5.6, color='#666666')

    fig.savefig(OUT_DIR / 'fig_benchmark_publication.pdf', bbox_inches='tight')
    fig.savefig(OUT_DIR / 'fig_benchmark_publication.png', dpi=300, bbox_inches='tight')
    print(f'Saved: {OUT_DIR / "fig_benchmark_publication.pdf"}')
    print(f'Saved: {OUT_DIR / "fig_benchmark_publication.png"}')

    # print the paired stats for the caption / log
    print('\n=== Paired Wilcoxon (gLV+AGORA better, one-sided) ===')
    for k in comp_keys:
        cv = np.array(data[k]['per_fold'])
        stat, p = wilcoxon(ref_fold, cv, alternative='less')
        wins = int((ref_fold < cv).sum())
        print(f'  vs {METHODS[k][0]:<16s}  wins {wins}/10  p={p:.4f}')


if __name__ == '__main__':
    main()
