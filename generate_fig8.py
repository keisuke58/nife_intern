#!/usr/bin/env python3
"""
generate_fig8.py — Fig 8: LOO A matrix stability (Section 3.5).

Best model: gLV α=0.25 (LOO-RMSE 0.0490), 10 folds.
Panels:
  A — Sign consistency heatmap (green=stable across folds)
  B — |A_map| vs LOO std scatter, coloured by regime
  C — Top 12 pairs: boxplot of A_ij distribution across folds
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.linewidth': 0.8,
})
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_net_flow_expanded import build_net_flow_expanded
from guild_replicator_dieckow import GUILD_ORDER, N_G

DIR     = HERE / 'results' / 'dieckow_cr'
OUT_DIR = HERE / 'results'
PATIENTS = list('ABCDEFGHKL')
GS = ['Actin.', 'Bacil.', 'Bact.', 'β-Prot.',
      'Clost.', 'Corio.', 'Fusob.', 'γ-Prot.', 'Negat.', 'Other']

C_ALIGN  = '#2166ac'
C_MUTED  = '#92c5de'
C_DATA   = '#d6604d'
C_OTHER  = '#b0b0b0'
REGIME_COLORS = {
    'aligned':    C_ALIGN,
    'muted':      C_MUTED,
    'data_driven':C_DATA,
    'other':      C_OTHER,
}

# ── Load data ──────────────────────────────────────────────────────────────
F_l12  = build_net_flow_expanded(use_agora=False, verbose=False)
F_full = build_net_flow_expanded(use_agora=True,  verbose=False)
nfs    = (np.abs(F_full) + np.abs(F_full.T)) / 2   # symmetric prior strength

A_loo = []
for fold in range(10):
    p = DIR / f'loo_glv_a0p25_fold{fold}.json'
    d = json.loads(p.read_text())
    A_loo.append(np.array(d['A']))

A_stack = np.array(A_loo)   # (10, N_G, N_G)
A_mean  = A_stack.mean(axis=0)
A_std   = A_stack.std(axis=0, ddof=1)
print(f'Loaded {len(A_loo)} LOO folds (gLV α=0.25)')

# ── Per-pair stats ─────────────────────────────────────────────────────────
pairs = [(i, j) for j in range(N_G) for i in range(j)]

def sign_consistency(vals):
    pos = (vals > 0).sum(); neg = (vals < 0).sum()
    return max(pos, neg) / len(vals)

rows = []
for i, j in pairs:
    vals = A_stack[:, i, j]
    sc   = sign_consistency(vals)
    a    = A_mean[i, j]
    f    = nfs[i, j]

    if f > 0 and np.sign(a) * np.sign(F_full[i,j]) > 0 and abs(a) > 0.25:
        reg = 'aligned'
    elif f > 0 and np.sign(a) * np.sign(F_full[i,j]) > 0 and abs(a) <= 0.1:
        reg = 'muted'
    elif f == 0 and abs(a) > 0.25:
        reg = 'data_driven'
    else:
        reg = 'other'

    rows.append({'i':i,'j':j,'A_mean':a,'A_std':A_std[i,j],'SC':sc,
                 'prior':f,'regime':reg,
                 'gi':GUILD_ORDER[i],'gj':GUILD_ORDER[j]})

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))

# ── Panel A: sign consistency heatmap ─────────────────────────────────────
ax = axes[0]
SC_mat = np.full((N_G, N_G), np.nan)
for r in rows:
    SC_mat[r['i'], r['j']] = r['SC']
    SC_mat[r['j'], r['i']] = r['SC']
np.fill_diagonal(SC_mat, 1.0)

im = ax.imshow(SC_mat, cmap='RdYlGn', vmin=0.5, vmax=1.0, aspect='auto')
fig.colorbar(im, ax=ax, fraction=0.038, pad=0.03, label='Sign consistency')

# outline pairs with prior
for r in rows:
    if r['prior'] > 0:
        for ri, rj in [(r['i'],r['j']),(r['j'],r['i'])]:
            ax.add_patch(plt.Rectangle((rj-0.5, ri-0.5), 1, 1,
                         fill=False, edgecolor='black', lw=1.0))

ax.set_xticks(range(N_G)); ax.set_xticklabels(GS, rotation=45, ha='right', fontsize=7.5)
ax.set_yticks(range(N_G)); ax.set_yticklabels(GS, fontsize=7.5)
ax.set_title('(A) LOO sign consistency\n(boxed = prior constrained)', fontsize=9, loc='left')

# ── Panel B: |A_mean| vs LOO std, coloured by regime ──────────────────────
ax2 = axes[1]
by_regime = {}
for r in rows:
    by_regime.setdefault(r['regime'], []).append(r)

regime_labels = {
    'aligned':     'Data+prior aligned',
    'muted':       'Prior-constrained, muted',
    'data_driven': 'Data-driven (no prior)',
    'other':       'Other',
}
for reg, grp in by_regime.items():
    xs = [abs(r['A_mean']) for r in grp]
    ys = [r['A_std'] for r in grp]
    ax2.scatter(xs, ys, c=REGIME_COLORS[reg], label=regime_labels[reg],
                s=30, alpha=0.8, zorder=3)
    # annotate top pairs
    for r in grp:
        if abs(r['A_mean']) > 0.8:
            ax2.annotate(f"{GS[r['i']]}↔{GS[r['j']]}",
                         (abs(r['A_mean']), r['A_std']),
                         fontsize=6.5, xytext=(3, 1),
                         textcoords='offset points', color='#333')

ax2.set_xlabel(r'$|\bar{A}_{ij}|$ (mean over folds)', fontsize=9)
ax2.set_ylabel(r'LOO std($A_{ij}$)', fontsize=9)
ax2.set_title('(B) Interaction strength vs LOO variability', fontsize=9, loc='left')
ax2.legend(fontsize=7.5, loc='upper left')

# ── Panel C: top 12 pairs boxplot ─────────────────────────────────────────
ax3 = axes[2]
top = sorted(rows, key=lambda r: abs(r['A_mean']), reverse=True)[:12]
top = sorted(top, key=lambda r: r['A_mean'])

box_data   = [A_stack[:, r['i'], r['j']] for r in top]
box_labels = [f"{GS[r['i']]}↔{GS[r['j']]}" for r in top]
box_colors = [REGIME_COLORS[r['regime']] for r in top]

bp = ax3.boxplot(box_data, vert=False, patch_artist=True,
                 medianprops={'color': 'k', 'lw': 1.5},
                 flierprops={'marker': 'o', 'ms': 3, 'alpha': 0.5})
for patch, col in zip(bp['boxes'], box_colors):
    patch.set_facecolor(col); patch.set_alpha(0.75)

ax3.set_yticks(range(1, len(top)+1))
ax3.set_yticklabels(box_labels, fontsize=7.5)
ax3.axvline(0, color='k', lw=0.6, ls='--')
ax3.set_xlabel(r'$A_{ij}$ across 10 LOO folds', fontsize=9)
ax3.set_title('(C) Top 12 pairs: fold-to-fold distribution', fontsize=9, loc='left')

# legend for panel C
handles = [mpatches.Patch(color=REGIME_COLORS[k], label=v, alpha=0.75)
           for k, v in regime_labels.items()]
ax3.legend(handles=handles, fontsize=7, loc='lower right')

fig.suptitle('Fig 8 — LOO A matrix stability (gLV α=0.25, 10-fold)', fontsize=10, y=1.01)
fig.tight_layout(pad=1.5)

for fmt in ('pdf', 'png'):
    p = OUT_DIR / f'fig8_loo_stability.{fmt}'
    fig.savefig(p, dpi=200, bbox_inches='tight')
    print(f'Saved: {p}')
plt.close(fig)

# ── Print summary for paper ───────────────────────────────────────────────
sc1 = sum(1 for r in rows if r['SC'] == 1.0)
sc9 = sum(1 for r in rows if r['SC'] >= 0.9)
print(f'\nSign consistency summary ({len(rows)} pairs):')
print(f'  SC = 1.0: {sc1} pairs')
print(f'  SC ≥ 0.9: {sc9} pairs')
print(f'  SC < 0.7: {sum(1 for r in rows if r["SC"] < 0.7)} pairs')
print('\nRegime counts:')
from collections import Counter
for k, v in Counter(r['regime'] for r in rows).most_common():
    print(f'  {regime_labels[k]}: {v}')
