#!/usr/bin/env python3
"""
analyse_loo_stability.py
  Load LOO fold A matrices → compute per-pair std, CV, sign consistency.
  Works with partial results (folds missing A are skipped).
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
})

_here = Path(__file__).resolve().parents[2]
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
    p = CR / f'loo_expanded_agora_a0p0_comb_g0p25_combined_fold{fold}.json'
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

REGIME_COLORS = {'data+prior aligned':    '#c0392b',
                 'prior-constrained muted': '#e67e22',
                 'data-driven no prior':    '#2980b9',
                 'variable':                '#95a5a6'}
REGIME_LABELS = {'data+prior aligned':    'Data+prior aligned (n=13)',
                 'prior-constrained muted': 'Prior-constrained muted (n=21)',
                 'data-driven no prior':    'Data-driven, no prior (n=10)',
                 'variable':                'Variable (n=1)'}

# Unambiguous 5-char abbreviations (Bacil ≠ Bctrd)
ABBR = ['Actin', 'Bacil', 'Bctrd', 'β-Pro', 'Clost', 'Corio', 'Fusob', 'γ-Pro', 'Negat', 'Other']

fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

# ── Panel A: Sign consistency heatmap ────────────────────────────────────────
ax1 = axes[0]
im1 = ax1.imshow(SC_mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
cbar = plt.colorbar(im1, ax=ax1, shrink=0.82, label='Sign consistency')
cbar.ax.tick_params(labelsize=8)

ax1.set_xticks(range(N_G)); ax1.set_yticks(range(N_G))
ax1.set_xticklabels(ABBR, fontsize=7.5, rotation=45, ha='right')
ax1.set_yticklabels(ABBR, fontsize=7.5)
ax1.set_title(f'(A) LOO sign consistency\n({n_folds}/10 folds; black border = sign prior)', fontsize=9)

# Mark pairs with AGORA/sign prior
for i, j in pairs:
    if nfs[i, j] != 0:
        for ri, rj in [(i,j),(j,i)]:
            ax1.add_patch(plt.Rectangle((rj-0.5, ri-0.5), 1, 1, fill=False,
                                         edgecolor='black', lw=1.4))

# ── Panel B: |A_map| vs std scatter (log x-scale) ────────────────────────────
ax2 = axes[1]
for regime_name, grp in df.groupby('regime'):
    ax2.scatter(grp.A_map.abs() + 1e-4, grp.A_std,   # +ε for log scale
                color=REGIME_COLORS[regime_name],
                label=REGIME_LABELS[regime_name],
                alpha=0.75, s=45, zorder=3,
                edgecolors='none')
    for _, r in grp[grp.A_map.abs() > 0.25].iterrows():
        lbl = f'{ABBR[int(r.i)]}↔{ABBR[int(r.j)]}'
        ax2.annotate(lbl, (abs(r.A_map) + 1e-4, r.A_std),
                     fontsize=6.5, xytext=(4, 2), textcoords='offset points')

ax2.set_xscale('log')
ax2.set_xlabel(r'$|A_{ij}|$ MAP estimate (all 10 patients, log scale)')
ax2.set_ylabel(r'Std across LOO folds')
ax2.set_title('(B) MAP magnitude vs LOO variability', fontsize=9)
ax2.legend(loc='upper left', framealpha=0.9, fontsize=7)
ax2.set_ylim(bottom=-0.003)

# ── Panel C: Boxplot of top 12 pairs by |A_map| ───────────────────────────────
ax3 = axes[2]
top_pairs = df.reindex(df.A_map.abs().nlargest(12).index).sort_values('A_map', ascending=True)

box_data, labels, box_colors = [], [], []
for _, r in top_pairs.iterrows():
    vals = A_stack[:, int(r.i), int(r.j)]
    box_data.append(vals)
    labels.append(f'{ABBR[int(r.i)]}↔{ABBR[int(r.j)]}')
    box_colors.append(REGIME_COLORS[r.regime])

bp = ax3.boxplot(box_data, vert=False, patch_artist=True,
                 medianprops={'color': 'k', 'lw': 1.5},
                 flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.5},
                 boxprops={'lw': 0.8}, whiskerprops={'lw': 0.8},
                 capprops={'lw': 0.8})
for patch, col in zip(bp['boxes'], box_colors):
    patch.set_facecolor(col); patch.set_alpha(0.65)

ax3.set_yticks(range(1, len(labels)+1))
ax3.set_yticklabels(labels, fontsize=8)
ax3.axvline(0, color='k', lw=0.7, ls='--')
ax3.set_xlabel(r'$A_{ij}$ across LOO folds')
ax3.set_title(f'(C) Top 12 pairs: LOO distribution\n({n_folds} folds)', fontsize=9)

# Regime colour legend on panel C
legend_patches = [mpatches.Patch(facecolor=REGIME_COLORS[k], alpha=0.7,
                                  label=REGIME_LABELS[k].split(' (')[0])
                  for k in REGIME_COLORS if k != 'variable']
ax3.legend(handles=legend_patches, loc='lower right', fontsize=6.5, framealpha=0.9)

fig.suptitle(r'$A$ matrix LOO stability  (AGORA $W$=1.0, all 10 folds)',
             fontsize=11, y=1.01)
fig.tight_layout()

for ext in ('png', 'pdf'):
    out = FIG / f'fig_loo_stability.{ext}'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f'Saved {out}')
