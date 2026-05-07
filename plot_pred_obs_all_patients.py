#!/usr/bin/env python3
"""
plot_pred_obs_all_patients.py
Generate per-patient pred vs observed stacked bar figure (all 10 patients).
Uses SS v2 fit (fit_hamilton_kegg_steadystate_v2.json) — numpy damped iteration,
no JAX required. Consistent with LOO-CV RMSE values in Result 5.
"""
import json
import numpy as np
from pathlib import Path

_here    = Path(__file__).resolve().parent
PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
FIT_FILE = _here / 'results' / 'dieckow_cr' / 'fit_hamilton_kegg_steadystate_v2.json'
OUT_DIR  = _here / 'results' / 'dieckow_cr'

# ── Load fit ──────────────────────────────────────────────────────────────────
d        = json.load(open(FIT_FILE))
A_np     = np.array(d['A'])            # (10,10)
b_all    = np.array(d['b_all'])        # (10,10)
PATIENTS = list('ABCDEFGHKL')
GUILDS   = ['Actinobacteria', 'Bacilli', 'Bacteroidia', 'Betaproteobacteria',
            'Clostridia', 'Coriobacteriia', 'Fusobacteriia',
            'Gammaproteobacteria', 'Negativicutes', 'Other']
phi_all  = np.load(PHI_NPY)           # (10, 3, 10)
n_sp     = len(GUILDS)
SS_ITER  = 2000
SS_DT    = 0.03

# ── LOO-CV RMSE values (from loo_ss_v2 results) ───────────────────────────────
LOO_RMSE = {
    'A': 0.084, 'B': 0.091, 'C': 0.118, 'D': 0.065,
    'E': 0.052, 'F': 0.082, 'G': 0.054, 'H': 0.176,
    'K': 0.088, 'L': 0.066,
}

# ── Numpy damped replicator iteration ─────────────────────────────────────────
def damped_iter(A, b, phi0, n_iter=SS_ITER, dt=SS_DT):
    phi = phi0.copy().astype(np.float64)
    for _ in range(n_iter):
        f   = b + A @ phi
        phi = phi * np.exp(dt * f)
        phi = np.maximum(phi, 0.0)
        s   = phi.sum()
        if s > 1e-10:
            phi /= s
    return phi

def predict_patient(A, b_p, phi0):
    phi2 = damped_iter(A, b_p, phi0)
    phi3 = damped_iter(A, b_p, phi2)
    return phi2, phi3

print('Computing SS v2 predictions (numpy)...', flush=True)
preds = []
for pi, pat in enumerate(PATIENTS):
    phi2, phi3 = predict_patient(A_np, b_all[pi], phi_all[pi, 0])
    preds.append((phi2, phi3))
    tr_rmse = float(np.sqrt((np.mean((phi2 - phi_all[pi,1])**2) +
                              np.mean((phi3 - phi_all[pi,2])**2)) / 2))
    print(f'  {pat}: train_RMSE={tr_rmse:.4f}  LOO_RMSE={LOO_RMSE[pat]:.3f}', flush=True)

# ── Color palette (10 guilds) ─────────────────────────────────────────────────
COLORS = [
    '#4C8DC4',  # Actinobacteria
    '#5BB85B',  # Bacilli
    '#E0873A',  # Bacteroidia
    '#9B59B6',  # Betaproteobacteria
    '#D62728',  # Clostridia
    '#17BECF',  # Coriobacteriia
    '#BCBD22',  # Fusobacteriia
    '#F7B6D2',  # Gammaproteobacteria
    '#8C6D31',  # Negativicutes
    '#AAAAAA',  # Other
]
SHORT = ['Actin.', 'Bacil.', 'Bact.', 'β-Prot.', 'Clost.',
         'Coriob.', 'Fusob.', 'γ-Prot.', 'Negat.', 'Other']

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   9,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, axes = plt.subplots(2, 5, figsize=(16, 7.2),
                          gridspec_kw={'hspace': 0.60, 'wspace': 0.15})

# x positions: week1obs | week2obs / week2pred | week3obs / week3pred
X_OBS  = [0.0, 1.15, 2.30]        # obs weeks 1,2,3
X_PRED = [1.55, 2.70]             # pred weeks 2,3
BAR_W  = 0.35

for pi, (pat, ax) in enumerate(zip(PATIENTS, axes.ravel())):
    phi_obs  = phi_all[pi]         # (3, n_sp)
    phi2p, phi3p = preds[pi]
    loo = LOO_RMSE[pat]

    # stacked bars — observed weeks 1,2,3
    bot_obs = np.zeros(3)
    for gi in range(n_sp):
        vals_obs = [phi_obs[0, gi], phi_obs[1, gi], phi_obs[2, gi]]
        ax.bar(X_OBS, vals_obs, width=BAR_W, bottom=bot_obs,
               color=COLORS[gi], edgecolor='none', linewidth=0)
        bot_obs += np.array(vals_obs)

    # stacked bars — predicted weeks 2,3 (hatched to distinguish)
    bot_pred = np.zeros(2)
    for gi in range(n_sp):
        vals_pred = [phi2p[gi], phi3p[gi]]
        ax.bar(X_PRED, vals_pred, width=BAR_W, bottom=bot_pred,
               color=COLORS[gi], edgecolor='white', linewidth=0.4,
               hatch='////', alpha=0.85)
        bot_pred += np.array(vals_pred)

    # separator lines between obs and pred
    for xsep in [0.97, 2.12]:
        ax.axvline(xsep, color='#bbb', lw=0.8, ls='--')

    ax.set_title(f'Patient {pat}  [LOO-RMSE {loo:.3f}]',
                 fontsize=9.5, fontweight='bold', pad=3)

    ax.set_xlim(-0.25, 3.10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0.0, 1.15, 1.55, 2.30, 2.70])
    ax.set_xticklabels(['W1\nobs', 'W2\nobs', 'W2\npred', 'W3\nobs', 'W3\npred'],
                        fontsize=7.5)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    if pi % 5 == 0:
        ax.set_yticklabels(['0', '', '0.5', '', '1'], fontsize=7.5)
        ax.set_ylabel('Guild fraction φ', fontsize=8)
    else:
        ax.set_yticklabels(['', '', '', '', ''])

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=COLORS[i], label=SHORT[i]) for i in range(n_sp)]
patches.append(mpatches.Patch(facecolor='grey', hatch='////', edgecolor='white',
                               alpha=0.85, label='predicted (////)'))
fig.legend(handles=patches, loc='lower center', ncol=11,
           fontsize=8, frameon=True, framealpha=0.9,
           bbox_to_anchor=(0.5, -0.04),
           title='Guild class  |  pattern = predicted', title_fontsize=8)

fig.suptitle(
    'Observed vs Predicted Guild Composition — all 10 patients\n'
    'SS v2 model (damped replicator, 34-pair prior, 2000 iter, Δt=0.03) · '
    'Mean LOO-RMSE = 0.0875  ·  Training fit shown',
    fontsize=10.5, fontweight='bold', y=1.01
)

for ext in ('png', 'pdf'):
    fig.savefig(OUT_DIR / f'fig_pred_obs_all_patients.{ext}',
                dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved fig_pred_obs_all_patients.*', flush=True)
