#!/usr/bin/env python3
"""
collect_loo_alpha.py — collect and compare LOO-CV results across alpha values.

Usage:
    python collect_loo_alpha.py                         # Hamilton model
    python collect_loo_alpha.py --model glv             # gLV model
    python collect_loo_alpha.py --model both --plot     # side-by-side comparison
    python collect_loo_alpha.py --alphas 0.0 0.25 0.5
"""
import argparse, json
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--alphas',    nargs='+', type=float, default=[0.0, 0.25, 0.5])
parser.add_argument('--model',     choices=['hamilton', 'glv', 'both'], default='hamilton')
parser.add_argument('--agora',     choices=['no', 'yes', 'both'], default='no',
                    help='AGORA layer: no=without, yes=with, both=side-by-side')
parser.add_argument('--plot',      action='store_true')
args = parser.parse_args()

HERE     = Path(__file__).parent
OUT_DIR  = HERE / 'results' / 'dieckow_cr'
PATIENTS = list('ABCDEFGHKL')

FILE_PATTERNS = {
    ('hamilton', False): 'loo_expanded_a{tag}_fold{fold}.json',
    ('hamilton', True):  'loo_expanded_agora_a{tag}_fold{fold}.json',
    ('glv',      False): 'loo_glv_a{tag}_fold{fold}.json',
    ('glv',      True):  'loo_glv_agora_a{tag}_fold{fold}.json',
}
BASELINE_PATTERNS = {
    'hamilton': 'loo_expanded_noprior_a0p0_fold{fold}.json',
    'glv':      'loo_glv_noprior_a0p0_fold{fold}.json',
}


def load_results(model, use_agora):
    results = {}
    for alpha in args.alphas:
        tag   = str(alpha).replace('.', 'p')
        folds = {}
        for fold in range(10):
            fname = OUT_DIR / FILE_PATTERNS[(model, use_agora)].format(tag=tag, fold=fold)
            if not fname.exists():
                continue
            d = json.load(open(fname))
            folds[fold] = {
                'patient':        PATIENTS[fold],
                'rmse':           d['rmse'],
                'sign_agreement': d.get('sign_agreement', '?'),
            }
        results[alpha] = folds
    return results


def load_baseline(model):
    folds = {}
    for fold in range(10):
        fname = OUT_DIR / BASELINE_PATTERNS[model].format(fold=fold)
        if not fname.exists():
            continue
        d = json.load(open(fname))
        folds[fold] = {'patient': PATIENTS[fold], 'rmse': d['rmse'],
                       'sign_agreement': d.get('sign_agreement', '?')}
    return folds


def print_table(results, model_label, baseline=None):
    n_cols = len(args.alphas) + (1 if baseline else 0)
    print(f'\n=== LOO-RMSE by alpha  [{model_label}] ===')
    print(f'{"Patient":>8s}', end='')
    if baseline:
        print(f'  {"no prior":>10s}   ', end='')
    for alpha in args.alphas:
        print(f'  alpha={alpha:<4g}', end='')
    print()
    print('-' * (9 + 13 * n_cols))

    all_rmse = {alpha: [] for alpha in args.alphas}
    base_vals = []
    for fold in range(10):
        print(f'{PATIENTS[fold]:>8s}', end='')
        if baseline:
            if fold in baseline:
                v = baseline[fold]['rmse']
                base_vals.append(v)
                print(f'  {v:10.4f}   ', end='')
            else:
                print(f'  {"—":>10s}   ', end='')
        for alpha in args.alphas:
            folds = results.get(alpha, {})
            if fold in folds:
                v = folds[fold]['rmse']
                all_rmse[alpha].append(v)
                print(f'  {v:10.4f}   ', end='')
            else:
                print(f'  {"—":>10s}   ', end='')
        print()

    print('-' * (9 + 13 * n_cols))
    print(f'{"mean":>8s}', end='')
    if baseline:
        print(f'  {np.mean(base_vals):10.4f}   ' if base_vals else f'  {"—":>10s}   ', end='')
    for alpha in args.alphas:
        vals = all_rmse[alpha]
        print(f'  {np.mean(vals):10.4f}   ' if vals else f'  {"—":>10s}   ', end='')
    print()
    print(f'{"std":>8s}', end='')
    if baseline:
        print(f'  {np.std(base_vals):10.4f}   ' if base_vals else f'  {"—":>10s}   ', end='')
    for alpha in args.alphas:
        vals = all_rmse[alpha]
        print(f'  {np.std(vals):10.4f}   ' if vals else f'  {"—":>10s}   ', end='')
    print()

    print(f'\n=== Sign Agreement  [{model_label}] ===')
    for alpha in args.alphas:
        folds = results.get(alpha, {})
        sas = [folds[f]['sign_agreement'] for f in sorted(folds)
               if 'sign_agreement' in folds[f]]
        print(f'  alpha={alpha}: {", ".join(sas)}')

    print()
    means = {a: np.mean(v) for a, v in all_rmse.items() if v}
    if means:
        best = min(means, key=means.get)
        print(f'Best alpha = {best}  (mean LOO-RMSE = {means[best]:.4f})')
    return all_rmse


# ── collect ───────────────────────────────────────────────────────────────────

models      = ['hamilton', 'glv'] if args.model == 'both' else [args.model]
agora_flags = [False, True] if args.agora == 'both' else [args.agora == 'yes']
all_results        = {}
all_rmse_by_model  = {}

for m in models:
    base = load_baseline(m)
    for ag in agora_flags:
        key   = (m, ag)
        label = f'{m.upper()} {"+ AGORA" if ag else "no AGORA"}'
        res   = load_results(m, ag)
        all_results[key]       = res
        all_rmse_by_model[key] = print_table(res, label, baseline=base)

# ── optional plot ─────────────────────────────────────────────────────────────

if args.plot:
    import matplotlib.pyplot as plt

    keys    = list(all_results.keys())
    n_rows  = len(keys)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows), squeeze=False)
    colors = ['#4C72B0', '#DD8452', '#55A868']

    for row, key in enumerate(keys):
        m, ag     = key
        results   = all_results[key]
        all_rmse  = all_rmse_by_model[key]
        label     = f'{m.upper()} {"+ AGORA" if ag else "no AGORA"}'

        # per-patient bar chart
        ax = axes[row, 0]
        x = np.arange(10)
        width = 0.8 / len(args.alphas)
        for k, alpha in enumerate(args.alphas):
            folds = results.get(alpha, {})
            vals = [folds[f]['rmse'] if f in folds else np.nan for f in range(10)]
            offset = (k - len(args.alphas) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width, label=f'α={alpha}',
                   color=colors[k % len(colors)], alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(PATIENTS)
        ax.set_xlabel('Patient')
        ax.set_ylabel('LOO-RMSE')
        ax.set_title(f'[{label}] Per-patient LOO-RMSE by α')
        ax.legend()

        # mean ± std vs alpha
        ax2 = axes[row, 1]
        alphas_done = [a for a in args.alphas if all_rmse[a]]
        means_v = [np.mean(all_rmse[a]) for a in alphas_done]
        stds_v  = [np.std(all_rmse[a])  for a in alphas_done]
        ax2.errorbar(alphas_done, means_v, yerr=stds_v, fmt='o-', capsize=5,
                     color='#4C72B0', linewidth=2, markersize=8)
        ax2.set_xlabel('competition_weight α')
        ax2.set_ylabel('mean LOO-RMSE')
        ax2.set_title(f'[{label}] Mean LOO-RMSE vs α')
        ax2.set_xticks(alphas_done)
        if alphas_done:
            best_i = int(np.argmin(means_v))
            ax2.axvline(alphas_done[best_i], color='red', linestyle='--', alpha=0.5,
                        label=f'best α={alphas_done[best_i]}')
            ax2.legend()

    plt.tight_layout()
    tag = '_'.join(f'{m}_{"agora" if ag else "noagora"}' for m, ag in keys)
    out_fig = OUT_DIR / f'loo_alpha_comparison_{tag}.png'
    plt.savefig(out_fig, dpi=150, bbox_inches='tight')
    print(f'Plot saved: {out_fig}')
    plt.show()
