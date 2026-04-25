#!/usr/bin/env python3
"""
Dieckow analysis plots 2-6:
  2. Structural data (CLSM) per patient/week
  3. A-matrix heatmaps (gLV best + Hamilton)
  4. Predicted vs observed trajectory (per-patient)
  5. Cross-prediction scatter W2 / W3
  6. A-matrix network diagram (signed edges)
"""

import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from guild_replicator_dieckow import (
    GUILD_ORDER, GUILD_COLORS, GUILD_SHORT, predict_trajectory
)
from load_structure_dieckow import load_structural_data, build_occupancy

ROOT    = Path(__file__).resolve().parent
OTU_DIR = ROOT / 'results' / 'dieckow_otu'
CR_DIR  = ROOT / 'results' / 'dieckow_cr'

PHI_NPY    = OTU_DIR / 'phi_guild_excel_class.npy'
FIT_GLV    = CR_DIR  / 'fit_guild_excel_class.json'
FIT_HAM    = CR_DIR  / 'fit_guild_hamilton.json'
STRUCT_XLS = ROOT / 'Datasets' / 'Abutment_Structure vs composition.xlsx'

phi_all = np.load(PHI_NPY)           # (n_p, 3, n_g)  — may be 8 patients
with open(FIT_GLV) as f:
    fit_glv = json.load(f)
with open(FIT_HAM) as f:
    fit_ham = json.load(f)

GUILDS   = fit_glv['guilds']
N_G      = len(GUILDS)
PATIENTS = fit_glv['patients']        # 8 patients for excel_class
N_P      = len(PATIENTS)
GCOLS    = [GUILD_COLORS.get(g, '#aaaaaa') for g in GUILDS]
SHORT    = [GUILD_SHORT.get(g, g) for g in GUILDS]
WEEK_EC  = ['#2266cc', '#cc5500', '#117733']

# ── filter phi to matching patients ──────────────────────────────────────────
all12 = list('ABCDEFGHIJKL')
phi10_idx = [all12.index(p) for p in list('ABCDEFGHKL') if p in all12]
# phi_all has 10 patients A B C D E F G H K L (I/J excluded)
pat10 = list('ABCDEFGHKL')
pat_idx = [pat10.index(p) for p in PATIENTS if p in pat10]
phi = phi_all[pat_idx, :, :N_G]      # (N_P, 3, N_G)

# ── 2. Structural data heatmaps ───────────────────────────────────────────────
print('Plot 2: Structural data...', flush=True)
struct = load_structural_data(STRUCT_XLS)
occ, max_occ = build_occupancy(struct, normalize=True)
all_pats = list('ABCDEFGHIJKL')

fig2, axes2 = plt.subplots(2, 2, figsize=(13, 8))
axes2 = axes2.flatten()
variables = ['VolumeLive', 'TotalArea', 'PerLive', 'VolumeTotal']
titles    = ['VolumeLive [µm³/µm²]', 'TotalArea [µm²]',
             'PerLive [%]',          'VolumeTotal [µm³/µm²]']

for ax, var, title in zip(axes2, variables, titles):
    mat = np.full((12, 3), np.nan)
    vdict = struct.get(var, {})
    for r, pat in enumerate(all_pats):
        for w in range(3):
            v = vdict.get((pat, w + 1))
            if v is not None:
                mat[r, w] = v
    im = ax.imshow(mat, aspect='auto', cmap='YlOrRd')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['W1', 'W2', 'W3'], fontsize=10)
    ax.set_yticks(range(12))
    ax.set_yticklabels(all_pats, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.8)
    for r in range(12):
        for c in range(3):
            if not np.isnan(mat[r, c]):
                ax.text(c, r, f'{mat[r,c]:.1f}', ha='center', va='center',
                        fontsize=6.5, color='white' if mat[r, c] > np.nanmax(mat) * 0.6 else 'black')

fig2.suptitle('CLSM structural data — 12 patients × 3 weeks', fontsize=13, fontweight='bold')
fig2.tight_layout()
fig2.savefig(OTU_DIR / 'structural_heatmap.pdf', bbox_inches='tight', dpi=150)
fig2.savefig(OTU_DIR / 'structural_heatmap.png', bbox_inches='tight', dpi=150)
print('  saved structural_heatmap', flush=True)

# occupancy line plot
fig2b, ax2b = plt.subplots(figsize=(10, 4))
occ_mat = np.full((12, 3), np.nan)
for r, pat in enumerate(all_pats):
    for w in range(3):
        v = occ.get((pat, w + 1))
        if v is not None:
            occ_mat[r, w] = v
for r in range(12):
    row = occ_mat[r]
    if not np.all(np.isnan(row)):
        ax2b.plot([1, 2, 3], row, marker='o', linewidth=1.5,
                  label=all_pats[r], alpha=0.85)
ax2b.set_xlabel('Week', fontsize=11)
ax2b.set_ylabel('Occupancy (norm.)', fontsize=11)
ax2b.set_title('Live biovolume occupancy = VolumeLive/TotalArea (normalised)', fontsize=11)
ax2b.set_xticks([1, 2, 3])
ax2b.legend(fontsize=8, ncol=4, loc='upper right')
ax2b.grid(alpha=0.3)
fig2b.tight_layout()
fig2b.savefig(OTU_DIR / 'structural_occupancy.pdf', bbox_inches='tight', dpi=150)
fig2b.savefig(OTU_DIR / 'structural_occupancy.png', bbox_inches='tight', dpi=150)
print('  saved structural_occupancy', flush=True)
plt.close('all')

# ── helper: reconstruct full A from upper triangle ────────────────────────────
def upper_to_full(A_upper_list, n):
    A = np.zeros((n, n))
    idx = 0
    for j in range(n):
        for i in range(j + 1):
            A[i, j] = A[j, i] = A_upper_list[idx]
            idx += 1
    return A

# ── 3. A-matrix heatmaps ──────────────────────────────────────────────────────
print('Plot 3: A-matrix heatmaps...', flush=True)
A_glv = np.array(fit_glv['A'])
A_ham = np.array(fit_ham['A'])
# A_glv is (11,11), A_ham is (10,10) — use their native sizes in heatmaps
N_G_glv = A_glv.shape[0]
N_G_ham = A_ham.shape[0]

fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(16, 6))
for ax, A, ng, title in [
    (ax3a, A_glv, N_G_glv, f'gLV fit  (RMSE={fit_glv["rmse"]:.4f})'),
    (ax3b, A_ham, N_G_ham, f'Hamilton fit  (RMSE={fit_ham["rmse"]:.4f})'),
]:
    short_ng = [GUILD_SHORT.get(g, g) for g in GUILD_ORDER[:ng]]
    vmax = max(abs(A).max(), 0.01)
    im = ax.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(ng)); ax.set_xticklabels(short_ng, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(ng)); ax.set_yticklabels(short_ng, fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, shrink=0.85)
    for i in range(ng):
        for j in range(ng):
            c = 'white' if abs(A[i, j]) > vmax * 0.6 else 'black'
            ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center', fontsize=6, color=c)

fig3.suptitle('A-matrix (guild interactions)', fontsize=13, fontweight='bold')
fig3.tight_layout()
fig3.savefig(CR_DIR / 'guild_Amatrix_comparison.pdf', bbox_inches='tight', dpi=150)
fig3.savefig(CR_DIR / 'guild_Amatrix_comparison.png', bbox_inches='tight', dpi=150)
print('  saved guild_Amatrix_comparison', flush=True)
plt.close('all')

# ── 4. Predicted vs observed trajectory ──────────────────────────────────────
print('Plot 4: Trajectory plots...', flush=True)
A_use   = A_glv
b_all   = np.array(fit_glv['b_all'])   # (N_P, N_G)

ncols = 4
nrows = (N_P + ncols - 1) // ncols
fig4, axes4 = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
axes4 = axes4.flatten()

for p_idx, pat in enumerate(PATIENTS):
    ax = axes4[p_idx]
    phi2_pred, phi3_pred = predict_trajectory(phi[p_idx, 0], b_all[p_idx], A_use)
    pred = np.stack([phi[p_idx, 0], phi2_pred, phi3_pred])  # (3, N_G)
    obs  = phi[p_idx]                                        # (3, N_G)

    for g in range(N_G):
        col = GCOLS[g]
        ax.plot([1, 2, 3], obs[:, g] * 100,  color=col, lw=2.0, marker='o', ms=5)
        ax.plot([1, 2, 3], pred[:, g] * 100, color=col, lw=1.5, ls='--', marker='s', ms=4, alpha=0.8)

    ax.set_title(f'Patient {pat}', fontsize=10, fontweight='bold')
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(['W1', 'W2', 'W3'], fontsize=9)
    ax.set_ylabel('%', fontsize=8)
    ax.set_ylim(-2, 100)
    ax.grid(alpha=0.25)

for i in range(N_P, len(axes4)):
    axes4[i].set_visible(False)

# legend
handles = [mpatches.Patch(facecolor=GCOLS[g], label=SHORT[g]) for g in range(N_G)]
handles += [
    plt.Line2D([0], [0], color='k', lw=2, marker='o', ms=5, label='Observed'),
    plt.Line2D([0], [0], color='k', lw=1.5, ls='--', marker='s', ms=4, label='Predicted'),
]
fig4.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01),
            fontsize=8, ncol=3, frameon=True)
fig4.suptitle(f'Predicted vs observed — gLV fit (RMSE={fit_glv["rmse"]:.4f})',
              fontsize=13, fontweight='bold', y=1.01)
fig4.tight_layout()
fig4.savefig(CR_DIR / 'guild_trajectory.pdf', bbox_inches='tight', dpi=150)
fig4.savefig(CR_DIR / 'guild_trajectory.png', bbox_inches='tight', dpi=150)
print('  saved guild_trajectory', flush=True)
plt.close('all')

# ── 5. Cross-prediction scatter ───────────────────────────────────────────────
print('Plot 5: Cross-prediction scatter...', flush=True)
obs_w2, pred_w2 = [], []
obs_w3, pred_w3 = [], []

for p_idx in range(N_P):
    phi2_pred, phi3_pred = predict_trajectory(phi[p_idx, 0], b_all[p_idx], A_use)
    obs_w2.extend(phi[p_idx, 1].tolist())
    pred_w2.extend(phi2_pred.tolist())
    obs_w3.extend(phi[p_idx, 2].tolist())
    pred_w3.extend(phi3_pred.tolist())

obs_w2  = np.array(obs_w2)  * 100
pred_w2 = np.array(pred_w2) * 100
obs_w3  = np.array(obs_w3)  * 100
pred_w3 = np.array(pred_w3) * 100

# per-guild colors for scatter
scatter_cols = GCOLS * N_P

fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, obs, pred, title in [
    (ax5a, obs_w2, pred_w2, 'W1 → W2 prediction'),
    (ax5b, obs_w3, pred_w3, 'W2 → W3 prediction'),
]:
    ax.scatter(obs, pred, c=scatter_cols, s=28, alpha=0.75, edgecolors='none')
    lim = max(obs.max(), pred.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', lw=1.2, alpha=0.5)
    r = float(np.corrcoef(obs, pred)[0, 1])
    rmse = float(np.sqrt(np.mean((obs - pred) ** 2)))
    ax.set_xlabel('Observed (%)', fontsize=11)
    ax.set_ylabel('Predicted (%)', fontsize=11)
    ax.set_title(f'{title}\nr={r:.3f}  RMSE={rmse:.3f}%', fontsize=11, fontweight='bold')
    ax.set_xlim(-1, lim); ax.set_ylim(-1, lim)
    ax.grid(alpha=0.25)

handles5 = [mpatches.Patch(facecolor=GCOLS[g], label=SHORT[g]) for g in range(N_G)]
fig5.legend(handles=handles5, loc='lower right', bbox_to_anchor=(0.99, 0.01),
            fontsize=9, ncol=2, frameon=True)
fig5.suptitle(f'Cross-prediction scatter — gLV (n={N_P} patients × {N_G} guilds)',
              fontsize=13, fontweight='bold')
fig5.tight_layout()
fig5.savefig(CR_DIR / 'guild_cross_prediction.pdf', bbox_inches='tight', dpi=150)
fig5.savefig(CR_DIR / 'guild_cross_prediction.png', bbox_inches='tight', dpi=150)
print('  saved guild_cross_prediction', flush=True)
plt.close('all')

# ── 6. A-matrix network diagram ──────────────────────────────────────────────
print('Plot 6: Network diagram...', flush=True)
import matplotlib.patheffects as pe

A = A_glv.copy()
np.fill_diagonal(A, 0)           # hide self-loops for readability
vmax = np.abs(A).max()
threshold = vmax * 0.08          # only draw edges above 8% of max

N = N_G
angles = np.linspace(0, 2 * np.pi, N, endpoint=False) - np.pi / 2
pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(N)}

fig6, ax6 = plt.subplots(figsize=(11, 11))
ax6.set_aspect('equal')
ax6.axis('off')

# draw edges
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        v = A[i, j]
        if abs(v) < threshold:
            continue
        x0, y0 = pos[j]
        x1, y1 = pos[i]
        color = '#d62728' if v > 0 else '#1f77b4'
        lw = 0.8 + 3.5 * abs(v) / vmax
        alpha = 0.45 + 0.5 * abs(v) / vmax
        ax6.annotate('', xy=(x1, y1), xytext=(x0, y0),
                     arrowprops=dict(arrowstyle='->', color=color,
                                     lw=lw, alpha=alpha,
                                     connectionstyle='arc3,rad=0.12'))

# draw nodes
node_r = 0.13
for i in range(N):
    x, y = pos[i]
    circ = plt.Circle((x, y), node_r, color=GCOLS[i], zorder=5, linewidth=1.5,
                       edgecolor='white')
    ax6.add_patch(circ)
    ax6.text(x * 1.3, y * 1.3, SHORT[i], ha='center', va='center',
             fontsize=9, fontweight='bold',
             path_effects=[pe.withStroke(linewidth=2.5, foreground='white')])

# legend
leg_handles = [
    mpatches.Patch(color='#d62728', label='Positive (promotion)'),
    mpatches.Patch(color='#1f77b4', label='Negative (inhibition)'),
]
ax6.legend(handles=leg_handles, loc='lower left', fontsize=10, frameon=True)
ax6.set_title(f'Guild interaction network — gLV A-matrix (|A_ij| ≥ {threshold:.3f})',
              fontsize=12, fontweight='bold', pad=12)
ax6.set_xlim(-1.7, 1.7); ax6.set_ylim(-1.7, 1.7)
fig6.tight_layout()
fig6.savefig(CR_DIR / 'guild_network.pdf', bbox_inches='tight', dpi=150)
fig6.savefig(CR_DIR / 'guild_network.png', bbox_inches='tight', dpi=150)
print('  saved guild_network', flush=True)
plt.close('all')

print('\nAll plots done.')
print(f'  OTU: {OTU_DIR}')
print(f'  CR:  {CR_DIR}')
