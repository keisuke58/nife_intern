# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_loo_rmse_comparison.py — LOO-RMSE comparison across models.

Reads per-fold JSONs for each model and plots mean ± std LOO-RMSE.
Models ordered from free to most regularised.

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_loo_rmse_comparison.py
"""

import json, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from thesis_style import use, clean_ax

ROOT   = pathlib.Path(__file__).parents[2]
LOODIR = ROOT / "results" / "dieckow_cr"
OUT    = ROOT / "results" / "fig_loo_rmse_comparison"

MODELS = [
    # (label, prefix, color)
    ('gLV free\n(no prior)',        'loo_glv_a0p0',              '#aec7e8'),
    ('gLV\n$\\alpha$=0.25',         'loo_glv_agora_a0p25',       '#7fb3d3'),
    ('Hamilton\n(no prior)',         'loo_expanded_a0p0',         '#c5e0b4'),
    ('Hamilton\nL1+L2 only',         'loo_expanded_a0p25',        '#5fad56'),
    ('Hamilton\nL1+L2+L3\n$W$=1.0', 'loo_expanded_agora_w1p0',  '#1a7431'),
]


def load_rmses(prefix):
    files = sorted(glob.glob(str(LOODIR / f"{prefix}_fold*.json")))
    if not files:
        return None
    return np.array([json.load(open(f))['rmse'] for f in files])


def main():
    fig, ax = plt.subplots(figsize=use(width_frac=0.7, aspect=0.75))

    means, stds, labels, colors = [], [], [], []
    for label, prefix, color in MODELS:
        rmses = load_rmses(prefix)
        if rmses is None:
            print(f'  skipped: {prefix}')
            continue
        means.append(rmses.mean())
        stds.append(rmses.std())
        labels.append(label)
        colors.append(color)

    x = np.arange(len(means))
    bars = ax.bar(x, means, 0.6, color=colors, edgecolor='white',
                  linewidth=0.5, yerr=stds, capsize=3,
                  error_kw={'elinewidth': 0.8, 'ecolor': '0.4'})

    # value labels
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.0008, f'{m:.4f}', ha='center', va='bottom',
                fontsize=6.5)

    # star on Hamilton L1+L2+L3 W=1.0 — find by label, not argmin
    # (argmin may land on gLV free due to numerical tie; the point is that
    #  Hamilton+AGORA *matches* gLV free despite added constraints)
    w1_idx = next((i for i, lbl in enumerate(labels) if 'W=1.0' in lbl),
                  len(means) - 1)
    ax.text(x[w1_idx], means[w1_idx] / 2, r'$\bigstar$',
            ha='center', va='center', fontsize=10, color='white')

    # annotation: "matches gLV free" with arrow pointing to the W=1.0 bar top
    ax.annotate('matches gLV free',
                xy=(x[w1_idx], means[w1_idx] + stds[w1_idx] + 0.0015),
                xytext=(x[w1_idx] - 1.0, means[w1_idx] + stds[w1_idx] + 0.009),
                fontsize=6.5, color='0.25',
                arrowprops=dict(arrowstyle='->', color='0.45', lw=0.7),
                ha='center')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('LOO-RMSE', fontsize=8)
    ax.set_ylim(0, max(means) * 1.25)
    clean_ax(ax)

    # divider between gLV and Hamilton
    n_glv = sum(1 for _, p, _ in MODELS if 'glv' in p and load_rmses(p) is not None)
    if n_glv > 0 and n_glv < len(means):
        ax.axvline(n_glv - 0.5, color='0.6', lw=0.7, ls='--')
        ax.text(n_glv/2 - 0.5, ax.get_ylim()[1]*0.97,
                'gLV', ha='center', va='top', fontsize=7.5, color='0.4')
        ax.text((n_glv + len(means))/2 - 0.5, ax.get_ylim()[1]*0.97,
                'Hamilton', ha='center', va='top', fontsize=7.5, color='0.4')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(f'{OUT}.{ext}')
        print(f'saved → {OUT}.{ext}')

    print('\nModel comparison:')
    for label, m, s in zip(labels, means, stds):
        flag = ' ← best' if m == min(means) else ''
        print(f'  {label.replace(chr(10)," "):30s}  {m:.4f} ± {s:.4f}{flag}')


if __name__ == '__main__':
    main()
