#!/usr/bin/env python3
"""
plot_ct_analysis.py — Two analyses:
  1. LOO-CV per-patient RMSE figure (fig15_loo_cv.pdf)
  2. CT1 vs CT2 stratified gLV A-matrix comparison (fig16_ct_Amatrix.pdf)

CT labels (KMeans k=2 on structural + W1 composition):
  CT1: E, G, K  (Bacilli-dominant)
  CT2: A, B, C, F, H  (diverse, Betaproteobacteria-elevated)

Outputs:
  docs/figures/dieckow/fig15_loo_cv.pdf
  docs/figures/dieckow/fig16_ct_Amatrix.pdf
"""

import json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
import pub_style; pub_style.apply()
from guild_replicator_dieckow import (
    GUILD_ORDER, N_G, GUILD_COLORS_LIST,
    default_A, pack, unpack, rmse_guild, predict_trajectory,
)

ROOT    = _here
CR_DIR  = ROOT / 'results' / 'dieckow_cr'
OTU_DIR = ROOT / 'results' / 'dieckow_otu'
FIGS_DOCS = Path(__file__).resolve().parents[1] / 'docs' / 'figures' / 'dieckow'
FIGS      = CR_DIR   # primary output: nife/results/dieckow_cr/

PHI_NPY  = OTU_DIR / 'phi_guild_excel_class.npy'
FIT_GLV  = CR_DIR  / 'fit_guild_excel_class.json'
LOO_GLV  = CR_DIR  / 'loo_cv_glv.json'
CT_JSON  = OTU_DIR / 'community_types.json'

SHORT = [g[:4] for g in GUILD_ORDER]
GCOLS = GUILD_COLORS_LIST

# ── Load data ────────────────────────────────────────────────────────────────
print('Loading data...', flush=True)

phi_all_10 = np.load(PHI_NPY)   # (10, 3, 11) — 10 patients A-L excluding I,J
PAT10 = list('ABCDEFGHKL')

fit_glv = json.loads(FIT_GLV.read_text())
PATIENTS = fit_glv['patients']   # 8 patients (subset of 10)
pat_idx  = [PAT10.index(p) for p in PATIENTS]
phi_all  = phi_all_10[pat_idx]   # (8, 3, 11)
N_P = len(PATIENTS)

A_glv   = np.array(fit_glv['A'])
b_all   = np.array(fit_glv['b_all'])

ct_data     = json.loads(CT_JSON.read_text())
patient_ct  = ct_data['patient_ct']   # dict: patient → 1 or 2

# LOO results
loo_glv = json.loads(LOO_GLV.read_text())
loo_per_patient = {d['patient']: d for d in loo_glv['per_patient']}

# CT labels for 8 patients
ct_labels = np.array([patient_ct[p] for p in PATIENTS])  # 1 or 2

CT1_idx = np.where(ct_labels == 1)[0]
CT2_idx = np.where(ct_labels == 2)[0]
CT1_pats = [PATIENTS[i] for i in CT1_idx]
CT2_pats = [PATIENTS[i] for i in CT2_idx]
print(f'CT1 patients: {CT1_pats}', flush=True)
print(f'CT2 patients: {CT2_pats}', flush=True)

# ── Figure 1: LOO-CV per-patient RMSE ────────────────────────────────────────
print('\nFig 1: LOO-CV figure...', flush=True)

fig, ax = plt.subplots(figsize=(8, 4.5))

held_rmse  = np.array([loo_per_patient[p]['rmse']       for p in PATIENTS])
train_rmse = np.array([loo_per_patient[p]['train_rmse'] for p in PATIENTS])

colors_bar = ['#2196F3' if ct_labels[i] == 1 else '#FF5722' for i in range(N_P)]
x = np.arange(N_P)
w = 0.35

bars_h = ax.bar(x - w/2, held_rmse  * 100, w, color=colors_bar, alpha=0.85,
                label='Held-out RMSE', edgecolor='white', linewidth=0.5)
bars_t = ax.bar(x + w/2, train_rmse * 100, w, color=colors_bar, alpha=0.40,
                label='Train RMSE (7 patients)', edgecolor='white', linewidth=0.5,
                hatch='//')

ax.axhline(loo_glv['loo_rmse_mean'] * 100, color='k', ls='--', lw=1.4,
           label=f'Mean LOO RMSE = {loo_glv["loo_rmse_mean"]*100:.2f}%')

ax.set_xticks(x)
ax.set_xticklabels([f'Pat {p}' for p in PATIENTS], fontsize=11)
ax.set_ylabel('RMSE (relative abundance, %)', fontsize=12)
ax.set_title('Leave-one-patient-out cross-validation — gLV+struct\n'
             '(solid = held-out patient; hatched = 7-patient train set)',
             fontsize=12, fontweight='bold')
ax.set_ylim(0, max(held_rmse.max(), train_rmse.max()) * 100 * 1.25)
ax.grid(axis='y', alpha=0.3)

ct1_patch = mpatches.Patch(color='#2196F3', label='CT1 (Bacilli-dominant): E, G, K')
ct2_patch = mpatches.Patch(color='#FF5722', label='CT2 (diverse): A, B, C, F, H')
ax.legend(handles=[bars_h, bars_t, ct1_patch, ct2_patch],
          fontsize=10, loc='upper right', frameon=True)

plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGS / f'fig15_loo_cv.{ext}', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS_DOCS / f'fig15_loo_cv.{ext}', dpi=300, bbox_inches='tight')
plt.close()
print('  saved fig15_loo_cv', flush=True)

# ── Figure 2: CT-stratified gLV fit ──────────────────────────────────────────
print('\nFig 2: CT-stratified A matrix...', flush=True)


def fit_glv_subset(phi_sub, n_starts=3, warm_A=None, lam=1e-4):
    """Fit gLV on a subset of patients. Returns A, b_all, rmse."""
    n_p_sub = phi_sub.shape[0]
    A0 = warm_A if warm_A is not None else default_A()
    b0 = np.zeros((n_p_sub, N_G))
    theta0 = pack(A0, b0)

    def loss(theta):
        A_loc, b_loc = unpack(theta, n_p_sub)
        return rmse_guild(A_loc, b_loc, phi_sub) + lam * np.sum(theta**2)

    best_loss = np.inf
    best_theta = theta0.copy()
    rng = np.random.default_rng(0)
    for s in range(n_starts):
        if s == 0:
            t0 = theta0.copy()
        else:
            t0 = theta0 + rng.normal(0, 0.05, size=theta0.shape)
        res = minimize(loss, t0, method='L-BFGS-B',
                       options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.fun < best_loss:
            best_loss = res.fun
            best_theta = res.x
        print(f'    start {s+1}/{n_starts}  loss={res.fun:.5f}', flush=True)

    A_fit, b_fit = unpack(best_theta, n_p_sub)
    rmse_val = rmse_guild(A_fit, b_fit, phi_sub)
    return A_fit, b_fit, rmse_val


print('  Fitting CT1 subset...', flush=True)
phi_ct1 = phi_all[CT1_idx]
A_ct1, b_ct1, rmse_ct1 = fit_glv_subset(phi_ct1, warm_A=A_glv.copy())
print(f'  CT1 RMSE = {rmse_ct1:.4f}', flush=True)

print('  Fitting CT2 subset...', flush=True)
phi_ct2 = phi_all[CT2_idx]
A_ct2, b_ct2, rmse_ct2 = fit_glv_subset(phi_ct2, warm_A=A_glv.copy())
print(f'  CT2 RMSE = {rmse_ct2:.4f}', flush=True)

# Difference matrix
A_diff = A_ct2 - A_ct1   # positive = stronger in CT2

# ── Plot: 3-panel (CT1 | CT2 | CT2 - CT1) ─────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

vmax_ct = max(np.abs(A_ct1).max(), np.abs(A_ct2).max())
vmax_d  = np.abs(A_diff).max()

def heatmap(ax, A, title, vmax, cmap):
    im = ax.imshow(A, cmap=cmap, vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(N_G)); ax.set_yticks(range(N_G))
    ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(SHORT, fontsize=7)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Source guild $j$', fontsize=10)
    ax.set_ylabel('Target guild $i$', fontsize=10)
    for i in range(N_G):
        for j in range(N_G):
            v = A[i, j]
            if abs(v) > 0.02 * vmax:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=5.5, color='white' if abs(v) > 0.5 * vmax else 'black')
    return im

im1 = heatmap(axes[0], A_ct1, f'CT1 A-matrix\n(n={len(CT1_idx)} patients: {", ".join(CT1_pats)})\nRMSE={rmse_ct1:.3f}',
              vmax_ct, 'RdBu_r')
im2 = heatmap(axes[1], A_ct2, f'CT2 A-matrix\n(n={len(CT2_idx)} patients: {", ".join(CT2_pats)})\nRMSE={rmse_ct2:.3f}',
              vmax_ct, 'RdBu_r')
im3 = heatmap(axes[2], A_diff, 'CT2 − CT1 (difference)\n(red = stronger in CT2, blue = stronger in CT1)',
              vmax_d, 'PiYG')

plt.colorbar(im1, ax=axes[0], shrink=0.75, label='$A_{ij}$')
plt.colorbar(im2, ax=axes[1], shrink=0.75, label='$A_{ij}$')
plt.colorbar(im3, ax=axes[2], shrink=0.75, label='$\\Delta A_{ij}$')

plt.suptitle('Community-type-stratified gLV interaction matrices\n'
             'CT1 = Bacilli-dominant (health-associated), CT2 = diverse (disease-enriched)',
             fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIGS / f'fig16_ct_Amatrix.{ext}', dpi=300, bbox_inches='tight')
    fig.savefig(FIGS_DOCS / f'fig16_ct_Amatrix.{ext}', dpi=300, bbox_inches='tight')
plt.close()
print('  saved fig16_ct_Amatrix', flush=True)

# ── Print top differences ─────────────────────────────────────────────────────
print('\nTop CT2 − CT1 A-matrix differences:', flush=True)
flat_diff = [(A_diff[i, j], GUILD_ORDER[i], GUILD_ORDER[j])
             for i in range(N_G) for j in range(N_G) if i != j]
flat_diff.sort(key=lambda x: abs(x[0]), reverse=True)
for val, tgt, src in flat_diff[:10]:
    sign = '+' if val > 0 else ''
    print(f'  {src[:12]:12s} → {tgt[:12]:12s}  ΔA = {sign}{val:.3f}', flush=True)

print('\nDone.')
