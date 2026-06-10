#!/usr/bin/env python3
"""
plot_benchmark_baselines.py — bar chart comparing LOO-CV RMSE:
  gLV variants vs. ML baselines (persistence, mean, Ridge, LASSO, RF).

Reads:  results/benchmark_baselines/rmse_table.json
Output: results/benchmark_baselines/fig_benchmark_baselines.pdf
        results/benchmark_baselines/fig_benchmark_baselines.png

Run (repo root, TeX Live on PATH for LaTeX rendering):
  PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
    python scripts/figures/plot_benchmark_baselines.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json
import numpy as np
import matplotlib.pyplot as plt
from thesis_style import use

_here = _pathlib.Path(__file__).resolve().parents[2]
IN_JSON = _here / 'results' / 'benchmark_baselines' / 'rmse_table.json'
OUT_DIR = _here / 'results' / 'benchmark_baselines'

# Display order and labels
METHODS = [
    ('gLV + AGORA',       r'gLV+AGORA',       '#1a6faf'),
    ('gLV (no prior)',    r'gLV (no prior)',   '#4c9ed9'),
    ('gLV + L1/L2 prior', r'gLV+L1/L2',       '#7fbfe8'),
    ('random_forest',     r'Random Forest',    '#e07b39'),
    ('mean_train',        r'Train mean',       '#b0b0b0'),
    ('persistence',       r'Persistence',      '#909090'),
    ('lasso',             r'LASSO',            '#d4b483'),
    ('ridge',             r'Ridge',            '#c4956a'),
]

DIVIDER_AFTER = 2   # draw vertical divider after gLV methods


def main():
    data = json.load(open(IN_JSON))

    means  = [data[k]['mean_rmse']  for k, _, _ in METHODS]
    stds   = [data[k]['std_rmse']   for k, _, _ in METHODS]
    labels = [lab for _, lab, _ in METHODS]
    colors = [col for _, _, col in METHODS]

    fig, ax = plt.subplots(figsize=use(width_frac=0.95, aspect=0.5))

    x = np.arange(len(METHODS))
    bars = ax.bar(x, means, color=colors, width=0.6, zorder=3,
                  edgecolor='white', linewidth=0.5)

    # error bars (±1 SD)
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='#333333',
                elinewidth=0.8, capsize=2.5, zorder=4)

    # divider between gLV and baselines
    div_x = DIVIDER_AFTER + 0.5
    ax.axvline(div_x, color='#444444', linewidth=0.8, linestyle='--', zorder=2)

    # bracket labels
    ax.text(1.0, max(means) * 0.97, 'mechanistic ODE',
            ha='center', va='top', fontsize=7, color='#1a6faf',
            transform=ax.get_xaxis_transform())
    ax.text(5.5, max(means) * 0.97, 'ML baselines',
            ha='center', va='top', fontsize=7, color='#777777',
            transform=ax.get_xaxis_transform())

    # value labels on top of each bar
    for xi, (mean, std) in enumerate(zip(means, stds)):
        ax.text(xi, mean + std + 0.002, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=6, color='#222222')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7.5)
    ax.set_ylabel('LOO-CV RMSE', fontsize=8)
    ax.set_ylim(0, max(means) * 1.22)
    ax.yaxis.grid(True, linewidth=0.4, color='#cccccc', zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

    fig.tight_layout()

    pdf = OUT_DIR / 'fig_benchmark_baselines.pdf'
    png = OUT_DIR / 'fig_benchmark_baselines.png'
    fig.savefig(pdf)
    fig.savefig(png, dpi=200)
    print(f'Saved: {pdf}')
    print(f'Saved: {png}')
    plt.close(fig)


if __name__ == '__main__':
    main()
