#!/usr/bin/env python3
"""
make_fitting_fig.py — publication-quality fitting results figure.

Shows per-patient stacked-bar compositions: observed (W1, W2, W3) vs
predicted (W2_pred, W3_pred) using AGORA W=1.0 full-data fit.

Output: figures/fig_fitting_results.png  (300 DPI)
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.integrate import solve_ivp

HERE   = Path(__file__).resolve().parent
CR     = HERE.parent / 'results' / 'dieckow_cr'
OTU    = HERE.parent / 'results' / 'dieckow_otu'
OUTDIR = HERE / 'figures'

FIT_FILE = CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'

GUILD_ORDER = [
    'Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
    'Clostridia', 'Coriobacteriia', 'Fusobacteriia', 'Gammaproteobacteria',
    'Negativicutes', 'Other',
]
G_SHORT = ['Actin.', 'Bacil.', 'Bact.', 'β-Prot.', 'Clost.',
           'Corio.', 'Fusob.', 'γ-Prot.', 'Negat.', 'Other']

GUILD_COLORS = [
    '#976832', '#20af53', '#ea2323', '#25b0e1',
    '#98c557', '#b69e7c', '#fac20b', '#156fb5',
    '#703a98', '#555555',
]

CT_MAP = {'A': 2, 'B': 2, 'C': 2, 'D': 1, 'E': 1,
          'F': 2, 'G': 1, 'H': 2, 'K': 1, 'L': 1}

plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        9,
    'axes.titlesize':   9,
    'axes.titleweight': 'bold',
    'axes.labelsize':   8,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'savefig.dpi':      300,
    'axes.spines.top':  False,
    'axes.spines.right':False,
})


def replicator_rhs(t, phi, b, A):
    f = b + A @ phi
    fmean = phi @ f
    return phi * (f - fmean)


def integrate_step(phi0, b, A, dt=1.0):
    sol = solve_ivp(replicator_rhs, [0, dt], phi0, args=(b, A),
                    method='RK45', rtol=1e-7, atol=1e-9)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0


d = json.load(open(FIT_FILE))
A      = np.array(d['A'])
b_all  = np.array(d['b_all'])
PATIENTS = d['patients']
phi_all  = np.load(OTU / 'phi_guild.npy')   # (10, 3, 10)

# Compute predictions
pred2, pred3 = [], []
rmse_list = []
for pi, pat in enumerate(PATIENTS):
    p2 = integrate_step(phi_all[pi, 0], b_all[pi], A)
    p3 = integrate_step(p2, b_all[pi], A)
    pred2.append(p2)
    pred3.append(p3)
    r = np.sqrt(np.mean((p2 - phi_all[pi, 1])**2 + (p3 - phi_all[pi, 2])**2) / 2)
    rmse_list.append(r)
    print(f'  {pat} RMSE={r:.4f}')

print(f'Mean RMSE: {np.mean(rmse_list):.4f}')

# ── Plot: 10 patients × 5 columns (W1obs, W2obs, W2pred, W3obs, W3pred) ───────
n_pat = len(PATIENTS)
fig, axes = plt.subplots(1, n_pat, figsize=(13.0, 3.8), sharey=True)
fig.suptitle('Observed vs Predicted Guild Compositions (AGORA W=1.0, training fit)',
             fontsize=10, fontweight='bold', y=1.01)

BAR_W = 0.18
POSITIONS = {'W1': 0.0, 'W2obs': 0.22, 'W2pred': 0.44, 'W3obs': 0.66, 'W3pred': 0.88}
LABELS = ['W1', 'W2\nobs', 'W2\npred', 'W3\nobs', 'W3\npred']
HATCH  = {'W2pred': '////', 'W3pred': '////'}

for pi, (pat, ax) in enumerate(zip(PATIENTS, axes)):
    data = {
        'W1':     phi_all[pi, 0],
        'W2obs':  phi_all[pi, 1],
        'W2pred': pred2[pi],
        'W3obs':  phi_all[pi, 2],
        'W3pred': pred3[pi],
    }
    for key, x in POSITIONS.items():
        bottom = 0.0
        for gi, (col, val) in enumerate(zip(GUILD_COLORS, data[key])):
            hatch = HATCH.get(key, None)
            ax.bar(x, val, bottom=bottom, width=BAR_W, color=col,
                   hatch=hatch, linewidth=0.3, edgecolor='white')
            bottom += val

    ax.set_xlim(-0.15, 1.05)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(list(POSITIONS.values()))
    ax.set_xticklabels(LABELS, fontsize=6.5)
    ct = CT_MAP.get(pat, 1)
    color = '#d6604d' if ct == 2 else '#2166ac'
    ax.set_title(f'{pat}\n(CT{ct})', fontsize=8, color=color)
    ax.text(0.5, -0.12, f'RMSE={rmse_list[pi]:.3f}', transform=ax.transAxes,
            ha='center', fontsize=6.5, color='#555555')
    if pi == 0:
        ax.set_ylabel('Relative abundance')
    else:
        ax.tick_params(left=False)

# Legend (guild colours)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, label=s)
                   for c, s in zip(GUILD_COLORS, G_SHORT)]
fig.legend(handles=legend_elements, loc='lower center', ncol=10,
           fontsize=7, bbox_to_anchor=(0.5, -0.14), frameon=False)

# Predicted bar pattern note
import matplotlib.lines as mlines
pred_note = mlines.Line2D([], [], color='#555555', marker='s', markersize=8,
                           markerfacecolor='white', markeredgecolor='#555555',
                           linestyle='None', label='hatched = predicted')
fig.legend(handles=[pred_note], loc='lower right', fontsize=7,
           bbox_to_anchor=(0.98, -0.12), frameon=False)

fig.tight_layout(rect=[0, 0.06, 1, 1])
out = OUTDIR / 'fig_fitting_results.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out.name}  ({out.stat().st_size//1024} KB)')
