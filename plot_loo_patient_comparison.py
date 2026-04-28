#!/usr/bin/env python3
"""Per-patient LOO-CV RMSE comparison: gLV vs gLV+KEGG-prior vs Hamilton+CLSM."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys_path = Path(__file__).parent
FIG_DIR = Path(__file__).parent.parent / 'docs' / 'figures' / 'dieckow'
RES = Path(__file__).parent / 'results' / 'dieckow_cr'

# CT labels (from paper)
CT1 = {'E', 'G', 'K'}
CT2 = {'A', 'B', 'C', 'F', 'H'}

def load(path):
    with open(path) as f:
        d = json.load(f)
    return {r['patient']: r['rmse'] for r in d['per_patient']}, d['loo_rmse_mean']

glv_rmse,      glv_mean      = load(RES / 'loo_cv_glv_pure.json')
kegg_rmse,     kegg_mean     = load(RES / 'loo_cv_glv_kegg_prior.json')
hamilton_rmse, hamilton_mean = load(RES / 'loo_cv_hamilton.json')

patients = ['A', 'B', 'C', 'E', 'F', 'G', 'H', 'K']

glv_vals      = [glv_rmse[p]      for p in patients]
kegg_vals     = [kegg_rmse[p]     for p in patients]
hamilton_vals = [hamilton_rmse[p] for p in patients]

x = np.arange(len(patients))
w = 0.25

# Publication style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})

fig, ax = plt.subplots(figsize=(7.0, 3.4))

c_glv      = '#4878CF'
c_kegg     = '#6ACC65'
c_hamilton = '#D65F5F'

b1 = ax.bar(x - w, glv_vals,      w, color=c_glv,      label=f'gLV (mean={glv_mean:.3f})',
            edgecolor='white', linewidth=0.5)
b2 = ax.bar(x,     kegg_vals,     w, color=c_kegg,     label=f'gLV+KEGG-prior (mean={kegg_mean:.3f})',
            edgecolor='white', linewidth=0.5)
b3 = ax.bar(x + w, hamilton_vals, w, color=c_hamilton, label=f'Hamilton+CLSM (mean={hamilton_mean:.3f})',
            edgecolor='white', linewidth=0.5)

# Annotate patient E (KEGG-prior outlier)
e_idx = patients.index('E')
ax.annotate(f'{kegg_vals[e_idx]:.3f}',
            xy=(e_idx, kegg_vals[e_idx]),
            xytext=(e_idx + 0.3, kegg_vals[e_idx] + 0.005),
            fontsize=7, color=c_kegg,
            arrowprops=dict(arrowstyle='->', color=c_kegg, lw=0.8))

# CT-type shading
for i, p in enumerate(patients):
    fc = '#FFF3CD' if p in CT2 else '#D4EDDA'
    ax.axvspan(i - 0.5, i + 0.5, color=fc, alpha=0.35, zorder=0)

# Mean lines
ax.axhline(glv_mean,      color=c_glv,      lw=1.0, ls='--', alpha=0.7)
ax.axhline(kegg_mean,     color=c_kegg,     lw=1.0, ls='--', alpha=0.7)
ax.axhline(hamilton_mean, color=c_hamilton, lw=1.0, ls='--', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels([f'Pat.~{p}' if False else p for p in patients])
ax.set_xlabel('Patient', fontsize=9)
ax.set_ylabel('LOO-CV RMSE', fontsize=9)
ax.set_title('Per-patient leave-one-out CV RMSE across three models', fontsize=9)

# CT legend patches
ct1_patch = mpatches.Patch(facecolor='#D4EDDA', alpha=0.6, label='CT1 (E, G, K)')
ct2_patch = mpatches.Patch(facecolor='#FFF3CD', alpha=0.6, label='CT2 (A, B, C, F, H)')

leg1 = ax.legend(handles=[b1, b2, b3], loc='upper left', fontsize=7.5, framealpha=0.85)
ax.add_artist(leg1)
ax.legend(handles=[ct1_patch, ct2_patch], loc='upper right', fontsize=7.5, framealpha=0.85)

ax.set_ylim(0, max(kegg_vals) * 1.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
FIG_DIR.mkdir(parents=True, exist_ok=True)
for ext in ('pdf', 'png'):
    fig.savefig(FIG_DIR / f'fig_loo_patient_comparison.{ext}', dpi=300, bbox_inches='tight')
print(f'Saved fig_loo_patient_comparison.pdf/png to {FIG_DIR}')
