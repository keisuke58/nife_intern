#!/usr/bin/env python3
"""Visualise the 10-guild A matrix and prediction vs observation."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from guild_replicator_dieckow import predict_trajectory, GUILD_ORDER, N_G

FIT_JSON = Path(__file__).parent / 'results' / 'dieckow_cr' / 'fit_guild.json'
PHI_NPY  = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'
OUT_DIR  = Path(__file__).parent / 'results' / 'dieckow_otu'
PATIENTS = list('ABCDEFGHKL')
SHORT    = [g[:5] for g in GUILD_ORDER]

with open(FIT_JSON) as f:
    fit = json.load(f)
A     = np.array(fit['A'])
b_all = np.array(fit['b_all'])
phi_obs = np.load(PHI_NPY)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ---- A matrix heatmap ----
ax = axes[0]
vmax = np.abs(A).max()
im = ax.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
ax.set_xticks(range(N_G)); ax.set_xticklabels(SHORT, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(N_G)); ax.set_yticklabels(SHORT, fontsize=8)
ax.set_xlabel('Source guild'); ax.set_ylabel('Target guild')
ax.set_title(f'10-guild interaction matrix A\n(RMSE={fit["rmse"]:.4f})', fontsize=10)
plt.colorbar(im, ax=ax, shrink=0.8)
for i in range(N_G):
    for j in range(N_G):
        ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center', fontsize=5,
                color='white' if abs(A[i,j]) > vmax*0.5 else 'black')

# ---- Obs vs Pred scatter (W2 + W3) ----
ax = axes[1]
all_obs, all_pred = [], []
for i in range(len(PATIENTS)):
    p2, p3 = predict_trajectory(phi_obs[i, 0], b_all[i], A)
    all_obs.extend(phi_obs[i, 1].tolist() + phi_obs[i, 2].tolist())
    all_pred.extend(p2.tolist() + p3.tolist())
all_obs  = np.array(all_obs)
all_pred = np.array(all_pred)
ax.scatter(all_obs, all_pred, s=10, alpha=0.4, color='steelblue')
lim = max(all_obs.max(), all_pred.max()) * 1.05
ax.plot([0, lim], [0, lim], 'r--', lw=1)
ax.set_xlabel('Observed φ'); ax.set_ylabel('Predicted φ')
ax.set_title('Observed vs predicted\n(W2 + W3, all patients)', fontsize=10)
corr = np.corrcoef(all_obs, all_pred)[0, 1]
ax.text(0.05, 0.92, f'r = {corr:.3f}', transform=ax.transAxes, fontsize=9)

fig.tight_layout()
for ext in ('pdf', 'png'):
    out = OUT_DIR / f'guild_Amatrix.{ext}'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    print(f'Saved: {out}')
plt.close(fig)
