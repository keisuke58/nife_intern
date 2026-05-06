#!/usr/bin/env python3
"""
analyse_loo_stability.py
  Load LOO fold A matrices → compute per-pair std, CV, sign consistency.
  Works with partial results (folds missing A are skipped).
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G

CR  = _here / 'results' / 'dieckow_cr'
FIG = CR / 'figs'
FIG.mkdir(exist_ok=True)

PATIENTS = list('ABCDEFGHKL')
GS = [g[:8] for g in GUILD_ORDER]

# Load MAP A (full fit, all patients)
d_map = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))
A_map = np.array(d_map['A'])
nf    = np.array(d_map['net_flow'])
nfs   = (nf + nf.T) / 2          # symmetrised prior flow

# Load LOO A matrices
A_loo = []
folds_used = []
for fold in range(10):
    p = CR / f'loo_expanded_agora_w1p0_fold{fold}.json'
    if not p.exists(): continue
    d = json.load(open(p))
    if 'A' not in d: continue
    A_loo.append(np.array(d['A']))
    folds_used.append(PATIENTS[fold])

A_stack = np.array(A_loo)   # (n_folds, N_G, N_G)
n_folds = len(A_loo)
print(f'LOO A matrices loaded: {n_folds} folds ({folds_used})')

# ── Per-pair statistics (upper triangle, off-diagonal) ───────────────────────
pairs = [(i, j) for j in range(N_G) for i in range(j)]   # i < j

A_mean = A_stack.mean(axis=0)
A_std  = A_stack.std(axis=0, ddof=1)

def sign_consistency(vals):
    """Fraction of folds where sign agrees with majority sign."""
    pos = (vals > 0).sum(); neg = (vals < 0).sum()
    return max(pos, neg) / len(vals)

stats_rows = []
for i, j in pairs:
    vals = A_stack[:, i, j]
    sc   = sign_consistency(vals)
    cv   = abs(A_std[i,j] / A_mean[i,j]) if abs(A_mean[i,j]) > 1e-6 else np.nan
    stats_rows.append({
        'i': i, 'j': j,
        'guild_i': GUILD_ORDER[i], 'guild_j': GUILD_ORDER[j],
        'A_map':  A_map[i, j],
        'A_mean': A_mean[i, j],
        'A_std':  A_std[i, j],
        'CV':     cv,
        'sign_consistency': sc,
        'prior': abs(nfs[i, j]),
    })

import pandas as pd
df = pd.DataFrame(stats_rows)
df.to_csv(CR / 'loo_stability_stats.csv', index=False)
print(f'Saved {CR}/loo_stability_stats.csv')

# ── Regime classification ──────────────────────────────────────────────────────
# Regime 1: Data+prior aligned — large |A_map|, high sign consistency, |prior|>0
# Regime 2: Prior-constrained muted — small |A_map|, high sign consistency, |prior|>0
# Regime 3: Data-driven no prior — any |A_map|, variable sign consistency, |prior|=0

def regime(row):
    if row['prior'] > 0 and row['sign_consistency'] >= 0.85 and abs(row['A_map']) > 0.15:
        return 'data+prior aligned'
    elif row['prior'] > 0 and row['sign_consistency'] >= 0.7 and abs(row['A_map']) <= 0.15:
        return 'prior-constrained muted'
    elif row['prior'] == 0:
        return 'data-driven no prior'
    else:
        return 'variable'

df['regime'] = df.apply(regime, axis=1)
print('\nRegime counts:')
print(df.regime.value_counts())

# ── Print top stable and unstable pairs ──────────────────────────────────────
print('\nTop stable pairs (sign_consistency=1.0, |A_map|>0.1):')
top = df[(df.sign_consistency == 1.0) & (df.A_map.abs() > 0.1)].sort_values('A_map', key=abs, ascending=False)
for _, r in top.head(10).iterrows():
    print(f'  {r.guild_i:20s}↔{r.guild_j:20s}  A={r.A_map:+.3f}  std={r.A_std:.3f}  SC={r.sign_consistency:.2f}  prior={r.prior:.1f}')

print('\nMost unstable pairs (sign_consistency<0.7, |A_map|>0.05):')
bot = df[(df.sign_consistency < 0.7) & (df.A_map.abs() > 0.05)].sort_values('sign_consistency')
for _, r in bot.head(10).iterrows():
    print(f'  {r.guild_i:20s}↔{r.guild_j:20s}  A={r.A_map:+.3f}  std={r.A_std:.3f}  SC={r.sign_consistency:.2f}  prior={r.prior:.1f}')

# ── Figure 1: Sign consistency heatmap ───────────────────────────────────────
SC_mat = np.full((N_G, N_G), np.nan)
for _, r in df.iterrows():
    SC_mat[int(r.i), int(r.j)] = r.sign_consistency
    SC_mat[int(r.j), int(r.i)] = r.sign_consistency

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
plt.rcParams.update({'font.size': 8})

ax1 = axes[0]
im1 = ax1.imshow(SC_mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
plt.colorbar(im1, ax=ax1, shrink=0.8, label='Sign consistency')
ax1.set_xticks(range(N_G)); ax1.set_yticks(range(N_G))
ax1.set_xticklabels(GS, fontsize=6.5, rotation=45, ha='right')
ax1.set_yticklabels(GS, fontsize=6.5)
ax1.set_title(f'LOO sign consistency ({n_folds}/10 folds)\n(green=stable, red=variable)', fontsize=8)
# Mark pairs with prior
for i, j in pairs:
    if nfs[i, j] != 0:
        for ri, rj in [(i,j),(j,i)]:
            ax1.add_patch(plt.Rectangle((rj-0.5, ri-0.5), 1, 1, fill=False,
                                         edgecolor='black', lw=1.2))

# ── Figure 2: A_map vs A_std scatter (stability) ─────────────────────────────
ax2 = axes[1]
REGIME_COLORS = {'data+prior aligned': '#e74c3c',
                 'prior-constrained muted': '#f39c12',
                 'data-driven no prior': '#2980b9',
                 'variable': '#95a5a6'}
for regime_name, grp in df.groupby('regime'):
    ax2.scatter(grp.A_map.abs(), grp.A_std, color=REGIME_COLORS[regime_name],
                label=regime_name, alpha=0.7, s=50, zorder=3)
    for _, r in grp[grp.A_map.abs() > 0.25].iterrows():
        ax2.annotate(f'{GS[int(r.i)]}↔{GS[int(r.j)]}',
                     (abs(r.A_map), r.A_std), fontsize=5.5,
                     xytext=(3,1), textcoords='offset points')
ax2.set_xlabel('|A_map| (MAP fit, all patients)', fontsize=8)
ax2.set_ylabel('Std across LOO folds', fontsize=8)
ax2.set_title('A stability: MAP magnitude vs LOO variability\n(regimes coloured)', fontsize=8)
ax2.legend(fontsize=7, loc='upper left')

# ── Figure 3: Per-pair boxplot for top ±8 MAP pairs ──────────────────────────
ax3 = axes[2]
top_pairs = df.reindex(df.A_map.abs().nlargest(12).index)
top_pairs = top_pairs.sort_values('A_map', ascending=True)

box_data = []
labels   = []
box_colors = []
for _, r in top_pairs.iterrows():
    vals = A_stack[:, int(r.i), int(r.j)]
    box_data.append(vals)
    labels.append(f'{GS[int(r.i)]}↔{GS[int(r.j)]}')
    box_colors.append('#e74c3c' if r.A_map > 0 else '#2980b9')

bp = ax3.boxplot(box_data, vert=False, patch_artist=True,
                  medianprops={'color': 'k', 'lw': 1.5})
for patch, col in zip(bp['boxes'], box_colors):
    patch.set_facecolor(col); patch.set_alpha(0.6)
ax3.set_yticks(range(1, len(labels)+1))
ax3.set_yticklabels(labels, fontsize=6.5)
ax3.axvline(0, color='k', lw=0.5, ls='--')
ax3.set_xlabel('A_ij across LOO folds', fontsize=8)
ax3.set_title(f'Top 12 pairs: LOO A distribution\n({n_folds} folds)', fontsize=8)

fig.suptitle(f'A matrix LOO stability (AGORA W=1.0, {n_folds}/10 folds)', fontsize=10)
fig.tight_layout()
out = FIG / 'fig_loo_stability.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'\nSaved {out}')
