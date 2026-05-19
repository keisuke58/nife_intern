#!/usr/bin/env python3
"""
collect_loo_results.py — Aggregate all LOO-CV JSON results and print comparison table.

Usage:
    python collect_loo_results.py
    python collect_loo_results.py --plot   # also generate figure
"""
import argparse, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--plot', action='store_true')
parser.add_argument('--dir', type=str,
                    default='/home/nishioka/IKM_Hiwi/nife/results/dieckow_cr')
args = parser.parse_args()

OUT = Path(args.dir)
files = sorted(list(OUT.glob('loo_expanded*.json')) + list(OUT.glob('loo_glv*.json')))

if not files:
    print('No LOO result files found.')
    raise SystemExit

PATIENTS = list('ABCDEFGHKL')

configs = defaultdict(list)

for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue

    rmse  = d.get('rmse', None)
    fold  = d.get('hold_idx', None)
    if rmse is None or fold is None or fold < 0:
        continue
    # skip old-format files that lack use_agora/alpha metadata
    if 'use_agora' not in d and 'alpha' not in d:
        continue

    agora  = d.get('use_agora', False)
    medium = d.get('agora_medium', 'v1') if agora else 'none'
    alpha  = d.get('alpha', 0.0)
    frac   = d.get('micom_fraction', 0.5) if medium == 'micom' else None
    noprior= d.get('no_prior', False)
    sa     = d.get('sign_agreement', '')
    tag    = d.get('model', '')

    is_glv = 'glv' in f.name
    model  = 'gLV' if is_glv else 'Hamilton'
    if noprior:
        key = f'{model} no-prior'
    elif not agora:
        key = f'{model} α={alpha}'
    elif medium == 'micom':
        key = f'{model} MICOM τ={frac} α={alpha}'
    else:
        key = f'{model} AGORA-{medium} α={alpha}'

    configs[key].append({'fold': fold, 'rmse': rmse, 'sa': sa})

# Deduplicate: for each (key, fold) keep only the entry with lowest RMSE
deduped = {}
for key, vals in configs.items():
    best_per_fold = {}
    for v in vals:
        f = v['fold']
        if f not in best_per_fold or v['rmse'] < best_per_fold[f]['rmse']:
            best_per_fold[f] = v
    deduped[key] = list(best_per_fold.values())
configs = deduped

print(f'{"Config":<35} {"N":>3} {"mean RMSE":>10} {"std":>7} {"min":>7} {"max":>7}  sign_agree')
print('-' * 90)

rows = []
for key in sorted(configs.keys()):
    vals = configs[key]
    rmses = [v['rmse'] for v in vals]
    n = len(rmses)
    mean = np.mean(rmses)
    std  = np.std(rmses)
    mn   = np.min(rmses)
    mx   = np.max(rmses)
    sa_all = [v['sa'] for v in vals if v['sa']]
    sa_str = sa_all[0] if sa_all else '—'
    rows.append((mean, key, n, std, mn, mx, sa_str))
    print(f'{key:<35} {n:>3} {mean:>10.4f} {std:>7.4f} {mn:>7.4f} {mx:>7.4f}  {sa_str}')

print()
if rows:
    best = min(rows, key=lambda r: r[0])
    print(f'Best: {best[1]}  mean={best[0]:.4f}')

# Per-fold breakdown for MICOM configs
print('\n=== Per-fold RMSE: MICOM configs ===')
for key in sorted(configs.keys()):
    if 'MICOM' not in key and 'micom' not in key.lower():
        continue
    vals = sorted(configs[key], key=lambda v: v['fold'])
    fold_str = '  '.join(f'{PATIENTS[v["fold"]]}={v["rmse"]:.4f}' for v in vals)
    print(f'  {key}: {fold_str}')

if args.plot:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Sort by mean RMSE
    rows_sorted = sorted(rows, key=lambda r: r[0])
    labels = [r[1] for r in rows_sorted]
    means  = [r[0] for r in rows_sorted]
    stds   = [r[3] for r in rows_sorted]

    fig, ax = plt.subplots(figsize=(max(8, len(labels)*0.8), 5))
    colors = ['#e05a4e' if 'MICOM' in l else '#4e7ce0' if 'AGORA' in l
              else '#aaaaaa' for l in labels]
    ax.bar(range(len(labels)), means, yerr=stds, color=colors,
           capsize=4, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('LOO-RMSE (mean ± std)')
    ax.set_title('Hamilton LOO-CV: sign prior comparison')
    ax.axhline(means[0], color='gray', lw=0.8, ls='--', alpha=0.5)

    import matplotlib.patches as mpatches
    legend = [mpatches.Patch(color='#e05a4e', label='MICOM'),
              mpatches.Patch(color='#4e7ce0', label='AGORA pFBA'),
              mpatches.Patch(color='#aaaaaa', label='L1+L2 / no-prior')]
    ax.legend(handles=legend, fontsize=8)
    fig.tight_layout()

    out_png = OUT / 'fig_loo_results_comparison.png'
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f'\nSaved {out_png}')
