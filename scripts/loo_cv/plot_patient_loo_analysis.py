#!/usr/bin/env python3
"""Per-patient LOO-CV analysis: community type vs prediction error."""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
RES  = HERE / 'results' / 'dieckow_cr'
sys.path.insert(0, str(HERE))
from guild_replicator_dieckow import GUILD_ORDER, GUILD_COLORS_LIST, N_G

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

phi = np.load(HERE / 'results' / 'dieckow_otu' / 'phi_guild.npy')
ct  = json.load(open(HERE / 'results' / 'dieckow_otu' / 'community_types.json'))

# LOO-CV results — corrected patient labels (phi indices 0-7 = ABCDEFGH)
PAT = list('ABCDEFGH')
folds = [json.load(open(RES / f'loo_hamilton_kegg_fold{i}.json')) for i in range(8)]
loo_rmse = [d['rmse'] for d in folds]

# CT assignment (from community_types.json, corrected)
pat_ct   = {p: ct['patient_ct'][p] for p in PAT}
ct_color = {1: '#4C8DC4', 2: '#e67e22'}   # CT1=blue, CT2=orange

fig, axes = plt.subplots(1, 3, figsize=(17, 5.5),
                          gridspec_kw={'width_ratios': [1.1, 2.0, 1.3]})
fig.subplots_adjust(left=0.05, right=0.97, top=0.87, bottom=0.22, wspace=0.38)

# ── Panel 1: LOO-RMSE bar chart colored by CT ────────────────────────────────
ax = axes[0]
colors = [ct_color[pat_ct[p]] for p in PAT]
bars = ax.bar(PAT, loo_rmse, color=colors, edgecolor='k', linewidth=0.6, width=0.6)
ax.axhline(np.mean(loo_rmse), color='k', ls='--', lw=1.0, alpha=0.5, label=f'Mean={np.mean(loo_rmse):.4f}')
ax.set_ylabel('LOO-CV RMSE')
ax.set_title('Hamilton + KEGG\nLOO-CV per patient', fontweight='bold')
ax.legend(fontsize=8)
for i, (p, r) in enumerate(zip(PAT, loo_rmse)):
    ax.text(i, r + 0.003, f'{r:.3f}', ha='center', va='bottom', fontsize=7.5)

from matplotlib.patches import Patch
legend_els = [Patch(fc=ct_color[1], ec='k', lw=0.4, label='CT1 (Bacilli-dominant)'),
              Patch(fc=ct_color[2], ec='k', lw=0.4, label='CT2 (Actinobacteria-moderate)')]
ax.legend(handles=legend_els, fontsize=8, loc='upper right')

# ── Panel 2: Guild composition wk1 for each patient ──────────────────────────
ax = axes[1]
x = np.arange(len(PAT)); w = 0.8
bottom = np.zeros(len(PAT))
for g in range(N_G):
    vals = [phi[i, 0, g] for i in range(8)]
    ax.bar(x, vals, w, bottom=bottom, color=GUILD_COLORS_LIST[g],
           label=GUILD_ORDER[g], edgecolor='none')
    bottom += np.array(vals)

ax.set_xticks(x); ax.set_xticklabels(PAT)
ax.set_ylabel('Fraction (week 1)')
ax.set_title('Community composition at week 1\n(color = guild)', fontweight='bold')
ax.set_ylim(0, 1.02)

# CT boundary lines
for i, p in enumerate(PAT):
    if pat_ct[p] == 1:
        ax.axvspan(i - 0.45, i + 0.45, alpha=0.08, color=ct_color[1], zorder=0)
    else:
        ax.axvspan(i - 0.45, i + 0.45, alpha=0.08, color=ct_color[2], zorder=0)

ax.legend(bbox_to_anchor=(0.5, -0.18), loc='upper center', fontsize=9,
          frameon=True, ncol=2)

# ── Panel 3: LOO-RMSE vs Actinobacteria fraction (covariate shift) ───────────
ax = axes[2]
actin_frac = [phi[i, 0, 0] for i in range(8)]   # guild 0 = Actinobacteria
sc = ax.scatter(actin_frac, loo_rmse,
                c=[ct_color[pat_ct[p]] for p in PAT],
                s=80, edgecolors='k', linewidth=0.7, zorder=5)
for p, x_p, y_p in zip(PAT, actin_frac, loo_rmse):
    ax.annotate(p, (x_p, y_p), textcoords='offset points', xytext=(5, 3), fontsize=9)

# regression line
z = np.polyfit(actin_frac, loo_rmse, 1)
xr = np.linspace(min(actin_frac)*0.9, max(actin_frac)*1.05, 50)
ax.plot(xr, np.polyval(z, xr), 'k--', lw=1.0, alpha=0.5)
r_val = float(np.corrcoef(actin_frac, loo_rmse)[0, 1])
ax.text(0.05, 0.93, f'r = {r_val:.2f}', transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#ccc', alpha=0.9))

ax.set_xlabel('Actinobacteria fraction (week 1)')
ax.set_ylabel('LOO-CV RMSE')
ax.set_title('Covariate shift:\nActinobacteria dominance → harder prediction', fontweight='bold')
ax.legend(handles=legend_els, fontsize=8)

fig.suptitle('Patient A as covariate-shift outlier — high Actinobacteria fraction unseen in training',
             fontsize=11, fontweight='bold', y=0.97)

for ext in ('pdf', 'png'):
    fig.savefig(RES / f'fig_patient_loo_analysis.{ext}', dpi=180, bbox_inches='tight')
plt.close(fig)
print('Saved fig_patient_loo_analysis.*')
