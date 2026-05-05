#!/usr/bin/env python3
"""Compare L1+L2 vs AGORA W=1.0 LOO-CV per patient."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

CR  = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_cr')
OTU = Path('/home/nishioka/IKM_Hiwi/nife/results/dieckow_otu')
FIG = CR / 'figs'
FIG.mkdir(exist_ok=True)

PATIENTS = list('ABCDEFGHKL')
ct_map = json.load(open(OTU / 'community_types.json'))['patient_ct']

# Load LOO summaries
l12_sum  = json.load(open(CR / 'loo_expanded_summary.json'))
agora_f  = CR / 'loo_expanded_agora_w1p0_summary.json'

if not agora_f.exists():
    print('AGORA LOO summary not found — run aggregate_loo_agora.py first')
    exit(1)

agora_sum = json.load(open(agora_f))

l12_pp   = l12_sum['per_patient']
agora_pp = agora_sum['per_patient']
common   = [p for p in PATIENTS if p in l12_pp and p in agora_pp]

l12_v   = [l12_pp[p]   for p in common]
agora_v = [agora_pp[p] for p in common]
delta   = [a - b for a, b in zip(agora_v, l12_v)]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
plt.rcParams.update({'font.size': 11})

x = np.arange(len(common))
w = 0.35
ct_colors = ['#1565C0' if ct_map[p] == 1 else '#B71C1C' for p in common]

# Panel 1: Grouped bar
ax = axes[0]
ax.bar(x - w/2, l12_v,   w, label='L1+L2 (34 pairs)', color='#1565C0', alpha=0.7)
ax.bar(x + w/2, agora_v, w, label='AGORA W=1.0 (35 pairs)', color='#2E7D32', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(common)
ax.set_ylabel('LOO-RMSE')
ax.set_title(f'LOO-RMSE per patient\n(L1+L2 mean={np.mean(l12_v):.4f}; AGORA mean={np.mean(agora_v):.4f})')
ax.legend(fontsize=9)
ax.axhline(np.mean(l12_v),   color='#1565C0', ls='--', lw=1, alpha=0.7)
ax.axhline(np.mean(agora_v), color='#2E7D32', ls='--', lw=1, alpha=0.7)
for i, p in enumerate(common):
    ax.text(i, max(l12_v[i], agora_v[i]) + 0.004, f'CT{ct_map[p]}',
            ha='center', fontsize=8, color=ct_colors[i])

# Panel 2: Scatter L1+L2 vs AGORA
ax = axes[1]
m = max(max(l12_v), max(agora_v)) * 1.05
ax.plot([0, m], [0, m], 'k--', lw=1, alpha=0.5, label='y = x')
for p, l, a in zip(common, l12_v, agora_v):
    c = '#1565C0' if ct_map[p] == 1 else '#B71C1C'
    ax.scatter(l, a, color=c, s=60, zorder=5)
    ax.annotate(p, (l, a), xytext=(4, 4), textcoords='offset points', fontsize=9)
ax.set_xlabel('L1+L2 LOO-RMSE')
ax.set_ylabel('AGORA W=1.0 LOO-RMSE')
ax.set_title('AGORA vs L1+L2 LOO-RMSE\n(below diagonal = AGORA better)')
ax.set_xlim(0, m); ax.set_ylim(0, m)
from matplotlib.patches import Patch
leg = [Patch(facecolor='#1565C0', label='CT1'), Patch(facecolor='#B71C1C', label='CT2')]
ax.legend(handles=leg, fontsize=9)

# Panel 3: Delta (AGORA - L1+L2)
ax = axes[2]
colors = ['#2E7D32' if d < 0 else '#c0392b' for d in delta]
ax.bar(x, delta, color=colors, alpha=0.8)
ax.axhline(0, color='k', lw=0.8)
ax.axhline(np.mean(delta), color='gray', ls='--', lw=1, label=f'Mean Δ={np.mean(delta):.4f}')
ax.set_xticks(x)
ax.set_xticklabels(common)
ax.set_ylabel('ΔLOO-RMSE (AGORA − L1+L2)')
ax.set_title('LOO-RMSE change: AGORA vs L1+L2\n(green = AGORA better, red = L1+L2 better)')
ax.legend(fontsize=9)

fig.suptitle(f'L1+L2 vs AGORA W=1.0 LOO-CV Comparison ({len(common)} patients)',
             fontsize=13, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = FIG / 'fig_loo_comparison_agora_vs_l12.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
print(f'\nL1+L2 LOO:       mean={np.mean(l12_v):.4f} ± {np.std(l12_v):.4f}')
print(f'AGORA W=1.0 LOO: mean={np.mean(agora_v):.4f} ± {np.std(agora_v):.4f}')
print(f'Mean Δ = {np.mean(delta):.4f} ({100*np.mean(delta)/np.mean(l12_v):+.1f}%)')
n_better = sum(1 for d in delta if d < 0)
print(f'AGORA better in {n_better}/{len(common)} patients')
