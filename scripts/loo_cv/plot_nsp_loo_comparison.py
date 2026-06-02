#!/usr/bin/env python3
"""NSP LOO-CV 全モデル比較図"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RES = Path(__file__).resolve().parents[2] / 'results' / 'dieckow_cr'
OUT = RES / 'fig_nsp_loo_comparison.pdf'

PAT_ALL = list('ABCDEFGHKL')

# ──────────────────────────────────────────────
# データ読み込み
# ──────────────────────────────────────────────

def load_folds(paths):
    folds = []
    for p in paths:
        p = Path(p)
        if not p.exists(): continue
        d = json.load(open(p))
        folds += d.get('per_patient', [])
    folds.sort(key=lambda x: x.get('fold', PAT_ALL.index(x.get('patient','A'))))
    return folds

# gLV free (8患者)
glv_d = json.load(open(RES / 'loo_cv_glv.json'))
glv_pp = {x['patient']: x['rmse'] for x in glv_d['per_patient']}

# gLV (all_models_bc.csv の行)
glv_rmse_all = 0.04549     # gLV free from all_models_bc
glv_bc_all   = 0.14615

# Hamilton expanded (best) — from all_models_bc
ham_rmse = 0.04924         # MacArthur c=0.05 が実質 best Hamilton
ham_bc   = 0.1559

# NSP IFT v2
nsp_v2_d = json.load(open(RES / 'loo_nsp_ift_v2_all10.json'))
nsp_v2_pp = {x['patient']: x['rmse'] for x in nsp_v2_d['per_patient']}
nsp_v2_rmses = list(nsp_v2_pp.values())

# NSP IFT v4
nsp_v4 = load_folds([RES/'loo_nsp_ift_v4_folds0-4.json', RES/'loo_nsp_ift_v4_folds5-9.json'])
nsp_v4_pp = {x['patient']: x['rmse'] for x in nsp_v4}

# NSP IFT v5
nsp_v5 = load_folds([RES/'loo_nsp_ift_v5_folds0-4.json', RES/'loo_nsp_ift_v5_folds5-9.json'])
nsp_v5_pp = {x['patient']: x['rmse'] for x in nsp_v5}

# NSP-φ (gLV replicator)
nsp_phi = load_folds([RES/'loo_nsp_phi_folds0-4.json', RES/'loo_nsp_phi_folds5-9.json'])
nsp_phi_pp = {x['patient']: x['rmse'] for x in nsp_phi}
nsp_phi_bc_pp = {x['patient']: x['bc'] for x in nsp_phi}

# ──────────────────────────────────────────────
# 図のスタイル
# ──────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3,
})

CMAP = {
    'gLV':        '#2196F3',
    'Hamilton':   '#4CAF50',
    'NSP v2':     '#FF9800',
    'NSP v4':     '#E91E63',
    'NSP v5':     '#9C27B0',
    'NSP-φ':      '#F44336',
}
ALPHA_BAR = 0.85

fig = plt.figure(figsize=(13, 8))
gs  = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
               left=0.08, right=0.97, top=0.93, bottom=0.09)

ax_bar  = fig.add_subplot(gs[0, 0])   # 左上: モデル比較 RMSE bars
ax_bc   = fig.add_subplot(gs[0, 1])   # 右上: BC bars
ax_pp   = fig.add_subplot(gs[1, 0])   # 左下: per-patient dots
ax_comp = fig.add_subplot(gs[1, 1])   # 右下: pred vs obs scatter

# ──────────────────────────────────────────────
# 1. Model summary bar chart (RMSE)
# ──────────────────────────────────────────────
models = [
    ('gLV',      glv_rmse_all,  np.std(list(glv_pp.values())),  CMAP['gLV']),
    ('Hamilton\n(MacArthur)', ham_rmse, 0.0,                    CMAP['Hamilton']),
    ('NSP IFT\nv2',  np.mean(nsp_v2_rmses), np.std(nsp_v2_rmses), CMAP['NSP v2']),
    ('NSP IFT\nv4',  np.mean(list(nsp_v4_pp.values())),
                     np.std(list(nsp_v4_pp.values())),           CMAP['NSP v4']),
    ('NSP IFT\nv5',  np.mean(list(nsp_v5_pp.values())),
                     np.std(list(nsp_v5_pp.values())),           CMAP['NSP v5']),
    ('NSP-φ\n(replicator)', np.mean(list(nsp_phi_pp.values())),
                            np.std(list(nsp_phi_pp.values())),   CMAP['NSP-φ']),
]
xs = np.arange(len(models))
bars = ax_bar.bar(xs, [m[1] for m in models], yerr=[m[2] for m in models],
                  color=[m[3] for m in models], alpha=ALPHA_BAR,
                  capsize=4, error_kw={'linewidth': 1.2}, width=0.6)
ax_bar.set_xticks(xs)
ax_bar.set_xticklabels([m[0] for m in models], fontsize=8.5)
ax_bar.set_ylabel('LOO-CV RMSE (W3)', fontsize=10)
ax_bar.set_title('Model comparison — RMSE', fontsize=11, fontweight='bold')
ax_bar.axhline(glv_rmse_all, color=CMAP['gLV'], lw=1, ls='--', alpha=0.5)
for i, (_, mu, _, col) in enumerate(models):
    ax_bar.text(i, mu + 0.006, f'{mu:.3f}', ha='center', va='bottom', fontsize=8, color='#333')
ax_bar.set_ylim(0, 0.28)
ax_bar.set_yticks(np.arange(0, 0.30, 0.05))

# ──────────────────────────────────────────────
# 2. BC bar chart
# ──────────────────────────────────────────────
v5_bcs  = [x['bc'] for x in nsp_v5]
phi_bcs = [x['bc'] for x in nsp_phi]

bc_models = [
    ('gLV',    glv_bc_all,            0,                     CMAP['gLV']),
    ('NSP v2', float('nan'),          0,                     CMAP['NSP v2']),
    ('NSP v5', np.mean(v5_bcs),       np.std(v5_bcs),        CMAP['NSP v5']),
    ('NSP-φ',  np.mean(phi_bcs),      np.std(phi_bcs),       CMAP['NSP-φ']),
]
xs2 = np.arange(len(bc_models))
for i, (lbl, mu, sd, col) in enumerate(bc_models):
    if np.isnan(mu):
        ax_bc.bar(i, 0, color=col, alpha=0.3, width=0.6)
        ax_bc.text(i, 0.01, 'n/a', ha='center', va='bottom', fontsize=8, color='gray')
    else:
        ax_bc.bar(i, mu, yerr=sd, color=col, alpha=ALPHA_BAR, capsize=4,
                  error_kw={'linewidth': 1.2}, width=0.6)
        ax_bc.text(i, mu + 0.015, f'{mu:.3f}', ha='center', va='bottom', fontsize=8, color='#333')
ax_bc.axhline(glv_bc_all, color=CMAP['gLV'], lw=1, ls='--', alpha=0.5)
ax_bc.set_xticks(xs2)
ax_bc.set_xticklabels([m[0] for m in bc_models], fontsize=9)
ax_bc.set_ylabel('Bray-Curtis dissimilarity', fontsize=10)
ax_bc.set_title('Model comparison — Bray-Curtis', fontsize=11, fontweight='bold')
ax_bc.set_ylim(0, 0.8)

# ──────────────────────────────────────────────
# 3. Per-patient dot plot (key models)
# ──────────────────────────────────────────────
pats_10 = PAT_ALL
pats_8  = [p for p in pats_10 if p in glv_pp]

xs_pp = np.arange(len(pats_10))

def pp_vals(pp_dict, pats):
    return [pp_dict.get(p, np.nan) for p in pats]

ax_pp.plot(xs_pp, pp_vals(nsp_phi_pp,  pats_10), 'o--', color=CMAP['NSP-φ'],
           alpha=0.7, ms=5, label='NSP-φ (replicator)', lw=1)
ax_pp.plot(xs_pp, pp_vals(nsp_v2_pp,   pats_10), 's-',  color=CMAP['NSP v2'],
           alpha=0.9, ms=6, label='NSP IFT v2', lw=1.5)
ax_pp.plot([pats_10.index(p) for p in pats_8],
           [glv_pp[p] for p in pats_8],
           '^-', color=CMAP['gLV'], alpha=0.9, ms=6, label='gLV', lw=1.5)
ax_pp.axhline(np.mean(nsp_v2_rmses), color=CMAP['NSP v2'], lw=1, ls=':', alpha=0.6)
ax_pp.axhline(glv_rmse_all,          color=CMAP['gLV'],    lw=1, ls=':', alpha=0.6)
ax_pp.set_xticks(xs_pp)
ax_pp.set_xticklabels(pats_10, fontsize=9)
ax_pp.set_xlabel('Patient', fontsize=10)
ax_pp.set_ylabel('LOO RMSE', fontsize=10)
ax_pp.set_title('Per-patient LOO RMSE', fontsize=11, fontweight='bold')
ax_pp.legend(fontsize=8, frameon=False)
ax_pp.set_ylim(0, 0.35)

# ──────────────────────────────────────────────
# 4. NSP v2 pred vs obs scatter (all 10 patients W3)
# ──────────────────────────────────────────────
# Load W3 obs and build pred from RMSE info — approximate via per-guild
# Use the log file if available, else skip
phi_npy = Path(__file__).resolve().parents[2] / 'results' / 'dieckow_otu' / 'phi_guild.npy'
if phi_npy.exists():
    phi_all = np.load(phi_npy)
    keep = phi_all[:, 0, :].sum(axis=1) > 1e-12
    phi_all = phi_all[keep]
    n_guild = phi_all.shape[2]

    # gLV vs NSP v2 RMSE scatter per patient
    pat_labels = pats_10
    glv_r = [glv_pp.get(p, np.nan) for p in pat_labels]
    nsp_r = [nsp_v2_pp.get(p, np.nan) for p in pat_labels]

    colors_sc = [CMAP['gLV'] if glv_pp.get(p, 1) < nsp_v2_pp.get(p, 1) else CMAP['NSP v2']
                 for p in pat_labels]

    for i, (g, n, p) in enumerate(zip(glv_r, nsp_r, pat_labels)):
        if np.isnan(g) or np.isnan(n): continue
        color = CMAP['gLV'] if g < n else CMAP['NSP v2']
        ax_comp.annotate('', xy=(g, n), xytext=(0.5*(g+g), 0.5*(n+n)))
        ax_comp.scatter(g, n, color=color, s=60, zorder=3)
        ax_comp.annotate(p, (g, n), textcoords='offset points',
                         xytext=(4, 3), fontsize=8, color='#444')

    lim = 0.22
    ax_comp.plot([0, lim], [0, lim], 'k--', lw=0.8, alpha=0.4)
    ax_comp.fill_between([0, lim], [0, 0], [0, lim], alpha=0.04, color=CMAP['gLV'])
    ax_comp.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.04, color=CMAP['NSP v2'])
    ax_comp.text(0.01, lim-0.02, 'gLV wins', fontsize=8, color=CMAP['gLV'], alpha=0.7)
    ax_comp.text(lim-0.10, 0.01, 'NSP wins', fontsize=8, color=CMAP['NSP v2'], alpha=0.7)
    ax_comp.set_xlim(0, lim); ax_comp.set_ylim(0, lim)
    ax_comp.set_xlabel('gLV RMSE', fontsize=10)
    ax_comp.set_ylabel('NSP IFT v2 RMSE', fontsize=10)
    ax_comp.set_title('gLV vs NSP per patient', fontsize=11, fontweight='bold')
    ax_comp.set_aspect('equal')

    glv_patch  = mpatches.Patch(color=CMAP['gLV'],    label='gLV better')
    nsp_patch  = mpatches.Patch(color=CMAP['NSP v2'], label='NSP better')
    ax_comp.legend(handles=[glv_patch, nsp_patch], fontsize=8, frameon=False)

# ──────────────────────────────────────────────
# 全体タイトル
# ──────────────────────────────────────────────
fig.suptitle('NSP Hamilton ODE LOO-CV — Dieckow 10-guild oral microbiome',
             fontsize=13, fontweight='bold', y=0.98)

fig.savefig(OUT, bbox_inches='tight', dpi=180)
fig.savefig(str(OUT).replace('.pdf', '.png'), bbox_inches='tight', dpi=180)
print(f'Saved: {OUT}')
