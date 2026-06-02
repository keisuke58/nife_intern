#!/usr/bin/env python3
"""
plot_b_A_clustering.py
  Fig 1: PCA of patient b vectors — do commensal/dysbiotic patients separate?
  Fig 2: A matrix heatmap across 10 LOO folds — which interactions are stable?

Usage:
    /home/nishioka/IKM_Hiwi/.venv_jax/bin/python plot_b_A_clustering.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
from guild_replicator_dieckow import GUILD_ORDER, N_G

CR       = HERE / 'results' / 'dieckow_cr'
PATIENTS = list('ABCDEFGHKL')

# Clinical metadata (from Dieckow dataset)
# CR = commensal remission, each patient's diagnosis
DIAGNOSIS = {
    'A': 'peri-implantitis', 'B': 'peri-implantitis',
    'C': 'healthy',          'D': 'healthy',
    'E': 'healthy',          'F': 'healthy',
    'G': 'peri-implantitis', 'H': 'peri-implantitis',
    'K': 'peri-implantitis', 'L': 'peri-implantitis',
}
DX_COLOR = {'healthy': '#4c78a8', 'peri-implantitis': '#e45756'}

# Full-fit b values
_full = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded.json'))
b_all = np.array(_full['b_all'])   # (10, 10)

# ════════════════════════════════════════════════════════════════════════════
# Fig 1: PCA of b vectors
# ════════════════════════════════════════════════════════════════════════════
scaler = StandardScaler()
b_s = scaler.fit_transform(b_all)
pca = PCA(n_components=2)
B2 = pca.fit_transform(b_s)
ev = pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(6.5, 5.5))
for i, pat in enumerate(PATIENTS):
    dx   = DIAGNOSIS[pat]
    col  = DX_COLOR[dx]
    mk   = 'o' if dx == 'healthy' else 's'
    ax.scatter(*B2[i], marker=mk, s=110, color=col, zorder=4,
               edgecolors='white', linewidths=0.8)
    ax.text(B2[i, 0] + 0.08, B2[i, 1] + 0.08, pat, fontsize=9,
            color=col, fontweight='bold')

# loadings
scale = np.abs(B2).max() * 0.55
for g, guild in enumerate(GUILD_ORDER):
    dx2 = pca.components_[0, g] * scale
    dy2 = pca.components_[1, g] * scale
    ax.annotate('', xy=(dx2, dy2), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='grey', lw=1.0))
    ax.text(dx2 * 1.18, dy2 * 1.18, guild[:5], fontsize=7,
            color='dimgrey', ha='center')

legend_handles = [
    mpatches.Patch(color=DX_COLOR['healthy'],          label='Healthy (●)'),
    mpatches.Patch(color=DX_COLOR['peri-implantitis'], label='Peri-implantitis (■)'),
]
ax.legend(handles=legend_handles, fontsize=9, loc='best')
ax.set_xlabel(f'PC1 ({ev[0]:.1f}%)', fontsize=11)
ax.set_ylabel(f'PC2 ({ev[1]:.1f}%)', fontsize=11)
ax.set_title('PCA of patient growth-rate vectors (b)', fontsize=12)
ax.axhline(0, color='lightgrey', lw=0.8); ax.axvline(0, color='lightgrey', lw=0.8)
ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(CR / 'fig_b_pca.png', dpi=150)
fig.savefig(CR / 'fig_b_pca.pdf')
print('Saved fig_b_pca.png')

# ════════════════════════════════════════════════════════════════════════════
# Fig 2: A matrix stability across LOO folds
# ════════════════════════════════════════════════════════════════════════════
A_folds = []
for fold in range(10):
    path = CR / f'loo_expanded_a0p0_fold{fold}.json'
    if not path.exists():
        continue
    d = json.load(open(path))
    A_folds.append(np.array(d['A']))

A_folds = np.array(A_folds)          # (10, 10, 10)
A_mean  = A_folds.mean(axis=0)
A_std   = A_folds.std(axis=0)
A_cv    = np.where(np.abs(A_mean) > 1e-6,
                   A_std / np.abs(A_mean), np.nan)  # coefficient of variation
# sign consistency: fraction of folds with same sign as mean
A_sign_mean = np.sign(A_mean)
sign_cons = np.mean(np.sign(A_folds) == A_sign_mean, axis=0)

short = [g[:5] for g in GUILD_ORDER]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel A: mean A
norm = TwoSlopeNorm(vmin=A_mean.min(), vcenter=0, vmax=A_mean.max())
im0 = axes[0].imshow(A_mean, cmap='RdBu_r', norm=norm)
axes[0].set_title('Mean A (10 LOO folds)', fontsize=11)
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Panel B: std A
im1 = axes[1].imshow(A_std, cmap='Oranges')
axes[1].set_title('Std A across folds', fontsize=11)
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# Panel C: sign consistency
im2 = axes[2].imshow(sign_cons, cmap='RdYlGn', vmin=0.5, vmax=1.0)
axes[2].set_title('Sign consistency (fraction)', fontsize=11)
plt.colorbar(im2, ax=axes[2], shrink=0.8)

for ax in axes:
    ax.set_xticks(range(N_G)); ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7)
    ax.set_yticks(range(N_G)); ax.set_yticklabels(short, fontsize=7)
    # annotate values
    for i in range(N_G):
        for j in range(N_G):
            pass  # keep clean

plt.suptitle('A matrix stability across LOO folds (Hamilton no-prior)', fontsize=12, y=1.01)
plt.tight_layout()
fig.savefig(CR / 'fig_A_stability.png', dpi=150, bbox_inches='tight')
fig.savefig(CR / 'fig_A_stability.pdf', bbox_inches='tight')
print('Saved fig_A_stability.png')

# ── Print most/least stable interactions ─────────────────────────────────────
print('\n=== Most stable interactions (sign_cons = 1.0) ===')
for i in range(N_G):
    for j in range(i, N_G):
        if sign_cons[i, j] == 1.0 and abs(A_mean[i, j]) > 0.01:
            sign_str = '+' if A_mean[i, j] > 0 else '-'
            print(f'  {GUILD_ORDER[i][:8]:8s} ↔ {GUILD_ORDER[j][:8]:8s}  '
                  f'mean={A_mean[i,j]:+.4f}  std={A_std[i,j]:.4f}  sign={sign_str}')

print('\n=== Least stable interactions (sign_cons < 0.8) ===')
for i in range(N_G):
    for j in range(i, N_G):
        if sign_cons[i, j] < 0.8:
            print(f'  {GUILD_ORDER[i][:8]:8s} ↔ {GUILD_ORDER[j][:8]:8s}  '
                  f'mean={A_mean[i,j]:+.4f}  std={A_std[i,j]:.4f}  '
                  f'sign_cons={sign_cons[i,j]:.2f}')

# b mean by diagnosis
print('\n=== Mean b by diagnosis ===')
h_idx  = [i for i, p in enumerate(PATIENTS) if DIAGNOSIS[p] == 'healthy']
pi_idx = [i for i, p in enumerate(PATIENTS) if DIAGNOSIS[p] == 'peri-implantitis']
for g, guild in enumerate(GUILD_ORDER):
    bh  = b_all[h_idx, g].mean()
    bpi = b_all[pi_idx, g].mean()
    diff = bpi - bh
    print(f'  {guild[:14]:14s}: healthy={bh:+.4f}  peri={bpi:+.4f}  Δ={diff:+.4f}')
