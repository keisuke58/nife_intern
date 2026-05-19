#!/usr/bin/env python3
"""Generate updated figures including MICOM results."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent / 'figures'
FIG_DIR.mkdir(exist_ok=True)

PATIENTS = list('ABCDEFGHKL')

# ── Data ──────────────────────────────────────────────────────────────────────
# Hamilton per-patient LOO-RMSE
ham_l1l2 = [0.0844, 0.0560, 0.0472, 0.0380, 0.0308, 0.0074, 0.0554, 0.0554, 0.0596, 0.0818]
ham_agora= [0.0799, 0.0545, 0.0473, 0.0338, 0.0305, 0.0078, 0.0527, 0.0546, 0.0606, 0.0807]
ham_micom= [0.0790, 0.0543, 0.0473, 0.0337, 0.0305, 0.0078, 0.0543, 0.0554, 0.0613, 0.0786]

glv_free =  [None]*10  # not per-patient available
glv_micom= [0.0652, 0.0512, 0.0505, 0.0346, 0.0210, 0.0357, 0.0534, 0.0466, 0.0785, 0.0763]
glv_a025 = [None]*10   # not stored per fold here

# Means
means = {
    'Hamilton\nno-prior':   0.0595,
    'Hamilton\nL1+L2':      0.0517,
    'Hamilton\nAGORA v1':   0.0562,
    'Hamilton\nMICOM τ=0.5':0.0502,
    'gLV\nno-prior':        0.0509,
    'gLV\nα=0.25':          0.0490,
    'gLV\nAGORA v1':        0.0499,
    'gLV\nMICOM τ=0.5':     0.0513,
}
stds = {
    'Hamilton\nno-prior':   0.024,
    'Hamilton\nL1+L2':      0.021,
    'Hamilton\nAGORA v1':   0.020,
    'Hamilton\nMICOM τ=0.5':0.021,
    'gLV\nno-prior':        0.018,
    'gLV\nα=0.25':          0.016,
    'gLV\nAGORA v1':        0.017,
    'gLV\nMICOM τ=0.5':     0.017,
}

# ── Figure 1: All-model LOO-RMSE bar chart ────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 4.5))

labels = list(means.keys())
vals   = [means[l] for l in labels]
errs   = [stds[l] for l in labels]

colors = []
for l in labels:
    if 'MICOM' in l and 'Hamilton' in l:  colors.append('#c0392b')
    elif 'MICOM' in l:                    colors.append('#e74c3c')
    elif 'AGORA' in l and 'Hamilton' in l:colors.append('#2980b9')
    elif 'AGORA' in l:                    colors.append('#5dade2')
    elif 'no-prior' in l:                 colors.append('#aaaaaa')
    elif 'L1+L2' in l:                    colors.append('#95a5a6')
    elif 'α=0.25' in l:                   colors.append('#27ae60')
    else:                                  colors.append('#aaaaaa')

x = np.arange(len(labels))
bars = ax.bar(x, vals, yerr=errs, color=colors, capsize=4,
              edgecolor='white', linewidth=0.5, width=0.7)

# Annotate best per model
for i, (v, e, c) in enumerate(zip(vals, errs, colors)):
    ax.text(i, v + e + 0.001, f'{v:.4f}', ha='center', va='bottom',
            fontsize=8, color='#333333')

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('LOO-RMSE (mean ± std)', fontsize=11)
ax.set_title('Leave-One-Out Cross-Validation: All Model Variants', fontsize=12, fontweight='bold')
ax.set_ylim(0, 0.080)

# Separator between Hamilton and gLV
ax.axvline(3.5, color='#cccccc', lw=1.5, ls='--')
ax.text(1.75, 0.077, 'Hamilton', ha='center', fontsize=10, color='#333333', style='italic')
ax.text(5.75, 0.077, 'gLV', ha='center', fontsize=10, color='#333333', style='italic')

legend = [
    mpatches.Patch(color='#c0392b', label='Hamilton + MICOM (best Hamilton)'),
    mpatches.Patch(color='#2980b9', label='Hamilton + AGORA v1'),
    mpatches.Patch(color='#27ae60', label='gLV α=0.25 (overall best)'),
    mpatches.Patch(color='#e74c3c', label='gLV + MICOM'),
    mpatches.Patch(color='#aaaaaa', label='No prior / L1+L2'),
]
ax.legend(handles=legend, fontsize=8, loc='upper right')
ax.grid(axis='y', alpha=0.3, lw=0.5)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig_loo_all_models_micom.pdf', bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig_loo_all_models_micom.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved fig_loo_all_models_micom')

# ── Figure 2: τ sensitivity ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 4))

taus   = [0.3,   0.4,   0.5]
rmses  = [0.0517, 0.0515, 0.0513]
sas    = [69,    72,    71]
colors_tau = ['#f39c12', '#e67e22', '#c0392b']

ax = axes[0]
ax.plot(taus, rmses, 'o-', color='#c0392b', lw=2, ms=8)
for t, r in zip(taus, rmses):
    ax.text(t, r + 0.0002, f'{r:.4f}', ha='center', va='bottom', fontsize=10)
ax.axhline(0.0490, color='#27ae60', lw=1.5, ls='--', label='gLV α=0.25 (overall best)')
ax.axhline(0.0499, color='#5dade2', lw=1.2, ls=':', label='gLV AGORA-v1')
ax.set_xlabel('MICOM tradeoff fraction τ', fontsize=11)
ax.set_ylabel('LOO-RMSE (mean)', fontsize=11)
ax.set_title('τ sensitivity: gLV + MICOM', fontsize=11, fontweight='bold')
ax.set_xlim(0.25, 0.55)
ax.set_ylim(0.048, 0.055)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
ax.set_xticks([0.3, 0.4, 0.5])

ax = axes[1]
ax.bar(taus, sas, width=0.07, color=colors_tau, edgecolor='white')
for t, s in zip(taus, sas):
    ax.text(t, s + 0.3, f'{s}/72', ha='center', va='bottom', fontsize=10)
ax.axhline(72, color='gray', lw=1, ls='--', alpha=0.5)
ax.set_xlabel('MICOM tradeoff fraction τ', fontsize=11)
ax.set_ylabel('Sign Agreement (of 72)', fontsize=11)
ax.set_title('τ sensitivity: Sign Agreement', fontsize=11, fontweight='bold')
ax.set_xlim(0.25, 0.55)
ax.set_ylim(65, 74)
ax.set_xticks([0.3, 0.4, 0.5])
ax.grid(axis='y', alpha=0.3)

fig.suptitle('MICOM Tradeoff Fraction τ Sensitivity (gLV model)', fontsize=12, y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig_tau_sensitivity.pdf', bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig_tau_sensitivity.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved fig_tau_sensitivity')

# ── Figure 3: Updated per-patient comparison (Hamilton: L1+L2 vs AGORA vs MICOM) ──
fig, ax = plt.subplots(figsize=(10, 4.5))

x = np.arange(10)
w = 0.25
b1 = ax.bar(x - w, ham_l1l2, w, label='L1+L2 only', color='#95a5a6', edgecolor='white')
b2 = ax.bar(x,     ham_agora, w, label='AGORA v1',   color='#2980b9', edgecolor='white')
b3 = ax.bar(x + w, ham_micom, w, label='MICOM τ=0.5',color='#c0392b', edgecolor='white')

ax.set_xticks(x)
ax.set_xticklabels(PATIENTS, fontsize=11)
ax.set_xlabel('Patient', fontsize=11)
ax.set_ylabel('LOO-RMSE', fontsize=11)
ax.set_title('Hamilton Model: Per-Patient LOO-RMSE (L1+L2 vs AGORA v1 vs MICOM)', fontsize=11, fontweight='bold')
ax.axhline(np.mean(ham_l1l2), color='#95a5a6', lw=1, ls='--', alpha=0.7)
ax.axhline(np.mean(ham_agora), color='#2980b9', lw=1, ls='--', alpha=0.7)
ax.axhline(np.mean(ham_micom), color='#c0392b', lw=1.5, ls='--', alpha=0.9)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
# Annotate means
ax.text(9.4, np.mean(ham_l1l2)+0.001, f'{np.mean(ham_l1l2):.4f}', fontsize=8, color='#95a5a6')
ax.text(9.4, np.mean(ham_agora)+0.001, f'{np.mean(ham_agora):.4f}', fontsize=8, color='#2980b9')
ax.text(9.4, np.mean(ham_micom)-0.003, f'{np.mean(ham_micom):.4f}', fontsize=8, color='#c0392b')
fig.tight_layout()
fig.savefig(FIG_DIR / 'fig_loo_per_patient_micom.pdf', bbox_inches='tight')
fig.savefig(FIG_DIR / 'fig_loo_per_patient_micom.png', dpi=200, bbox_inches='tight')
plt.close()
print('Saved fig_loo_per_patient_micom')
