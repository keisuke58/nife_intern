# [nife-pathshim]
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parents[2]))

"""fig_w_sweep_thesis.py — W sweep phase-transition figure for thesis.

Shows how increasing metabolic-prior weight W affects sign agreement (%)
and training RMSE. For W=1.0 the mean LOO-RMSE is also shown as a dotted
horizontal reference line.

Run:
    PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH \
        python scripts/figures/fig_w_sweep_thesis.py
"""

import json, glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from thesis_style import use, clean_ax, tight_ylim, PALETTE

ROOT   = pathlib.Path(__file__).parents[2]
FITDIR = ROOT / "results" / "dieckow_cr"
OUT    = ROOT / "results" / "fig_w_sweep_thesis"

W_VALUES = [0.5, 1.0, 1.5, 2.0]
W_LABELS = ['0.5', '1.0', '1.5', '2.0']
W_KEYS   = ['w0p5', 'w1p0', 'w1p5', 'w2p0']

COL_SA   = PALETTE["health"]
COL_RMSE = PALETTE["dysbiosis"]


def parse_sign_pct(s):
    """Convert '70/70' -> 100.0 or float -> percent."""
    if isinstance(s, str) and '/' in s:
        num, den = s.split('/')
        return 100.0 * int(num) / int(den)
    v = float(s)
    return v * 100.0 if v <= 1.0 else v


def load_fit(w_key):
    fname = FITDIR / ('fit_glv_hamilton_kegg_expanded_agora_' + w_key + '.json')
    return json.load(open(fname))


def load_loo_w1p0():
    files = sorted(glob.glob(str(FITDIR / 'loo_expanded_agora_w1p0_fold*.json')))
    if not files:
        return None, None
    rmses = np.array([json.load(open(f))['rmse'] for f in files])
    return rmses.mean(), rmses.std()


def main():
    fits      = [load_fit(k) for k in W_KEYS]
    sign_pcts = [parse_sign_pct(f['sign_agreement']) for f in fits]
    train_rm  = [f['rmse'] for f in fits]

    loo_mean, loo_std = load_loo_w1p0()

    # use integer index positions so spacing is uniform for discrete W values
    xpos = np.arange(len(W_VALUES))
    w1_idx = W_KEYS.index('w1p0')

    fig, ax1 = plt.subplots(figsize=use(width_frac=0.6, aspect=0.75))
    ax2 = ax1.twinx()

    # sign agreement — left axis
    ax1.plot(xpos, sign_pcts, 'o-', color=COL_SA, lw=1.4, ms=5,
             label='Sign agreement')

    # train RMSE — right axis
    ax2.plot(xpos, train_rm, 's--', color=COL_RMSE, lw=1.2, ms=4,
             label='Train RMSE')

    # LOO RMSE reference at W=1.0
    if loo_mean is not None:
        ax2.axhline(loo_mean, color=COL_RMSE, lw=0.8, ls=':', alpha=0.7)
        ax2.text(len(W_VALUES) - 0.95, loo_mean + 0.0005,
                 'mean LOO', fontsize=6, color=COL_RMSE, va='bottom')

    # W=1.0 vertical dashed line
    ax1.axvline(w1_idx, color='0.45', lw=0.75, ls='--', zorder=0)

    # axis decoration
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([r'$W=' + lbl + r'$' for lbl in W_LABELS], fontsize=8)
    ax1.set_xlabel(r'AGORA prior weight $W$', fontsize=9)
    ax1.set_ylabel(r'Sign agreement (\%)', color=COL_SA, fontsize=8)
    ax2.set_ylabel('RMSE', color=COL_RMSE, fontsize=8)
    ax1.tick_params(axis='y', labelcolor=COL_SA, labelsize=7)
    ax2.tick_params(axis='y', labelcolor=COL_RMSE, labelsize=7)

    tight_ylim(ax1, sign_pcts, margin=0.18)
    tight_ylim(ax2, train_rm + ([loo_mean] if loo_mean else []), margin=0.15)

    clean_ax(ax1)
    ax2.spines['top'].set_visible(False)

    handles = [
        mlines.Line2D([], [], color=COL_SA,   marker='o', lw=1.2, label='Sign agreement'),
        mlines.Line2D([], [], color=COL_RMSE, marker='s', lw=1.0, ls='--', label='Train RMSE'),
    ]
    ax1.legend(handles=handles, fontsize=7, loc='lower left',
               framealpha=0.9, edgecolor='0.8')

    fig.tight_layout()
    for ext in ('pdf', 'png'):
        fig.savefig(str(OUT) + '.' + ext)
        print('saved ->', str(OUT) + '.' + ext)

    print('\nW-sweep summary:')
    for w, sa, tr in zip(W_LABELS, sign_pcts, train_rm):
        print('  W=' + w + '  sign=' + str(round(sa, 1)) + '%  train_rmse=' + str(round(tr, 5)))
    if loo_mean is not None:
        print('  LOO RMSE W=1.0: ' + str(round(loo_mean, 5)) + ' +/- ' + str(round(loo_std, 5)))


if __name__ == '__main__':
    main()
