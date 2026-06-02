#!/usr/bin/env python3
"""
plot_agora_w1p0_results.py
  1. A matrix heatmap (W=1.0 vs L1+L2 old)
  2. b_p CT1 vs CT2 comparison
  3. Sign-agreement change visualization
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER
GS = [g[:6] for g in GUILD_ORDER]

CR  = _here / 'results' / 'dieckow_cr'
OTU = _here / 'results' / 'dieckow_otu'
FIG = _here / 'results' / 'dieckow_cr' / 'figs'
FIG.mkdir(exist_ok=True)

# ── Load fits ──────────────────────────────────────────────────────────────
d_old  = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded.json'))
d_new  = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))

A_old  = np.array(d_old['A'])
A_new  = np.array(d_new['A'])
b_old  = np.array(d_old['b_all'])   # (10, 10)
b_new  = np.array(d_new['b_all'])
PATIENTS = d_new['patients']         # ABCDEFGHKL

# ── CT labels ─────────────────────────────────────────────────────────────
ct_map  = json.load(open(OTU / 'community_types.json'))['patient_ct']
ct_arr  = np.array([ct_map[p] for p in PATIENTS])  # 1 or 2
ct1_idx = np.where(ct_arr == 1)[0]
ct2_idx = np.where(ct_arr == 2)[0]
ct1_pat = [PATIENTS[i] for i in ct1_idx]
ct2_pat = [PATIENTS[i] for i in ct2_idx]

# ── Figure layout ─────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
gs  = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.4)
ax_aold = fig.add_subplot(gs[0, 0])
ax_anew = fig.add_subplot(gs[0, 1])
ax_diff = fig.add_subplot(gs[0, 2])
ax_bp1  = fig.add_subplot(gs[1, 0])
ax_bp2  = fig.add_subplot(gs[1, 1])
ax_sa   = fig.add_subplot(gs[1, 2])

plt.rcParams.update({'font.size': 8})
CLIM = 0.5

# ── A matrix heatmaps ─────────────────────────────────────────────────────
def heatmap(ax, A, title, clim=CLIM):
    im = ax.imshow(A, cmap='RdBu_r', vmin=-clim, vmax=clim, aspect='auto')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(GS, fontsize=6, rotation=45, ha='right')
    ax.set_yticklabels(GS, fontsize=6)
    ax.set_title(title, fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.7)

heatmap(ax_aold, A_old, f'A (L1+L2)\nRMSE={d_old["rmse"]:.4f} SA={d_old["sign_agreement"]}')
heatmap(ax_anew, A_new, f'A (AGORA W=1.0)\nRMSE={d_new["rmse"]:.4f} SA={d_new["sign_agreement"]}')
heatmap(ax_diff, A_new - A_old, 'ΔA (new − old)', clim=0.2)

# ── b_p: CT1 vs CT2 per guild ─────────────────────────────────────────────
x = np.arange(10)
w = 0.35
for ax, b_mat, label in [(ax_bp1, b_old, 'b̂  L1+L2'), (ax_bp2, b_new, 'b̂  AGORA W=1.0')]:
    b_ct1 = b_mat[ct1_idx]   # (5,10)
    b_ct2 = b_mat[ct2_idx]   # (5,10)
    ax.bar(x - w/2, b_ct1.mean(0), w, yerr=b_ct1.std(0), label='CT1', color='#1565C0', alpha=0.8, capsize=3)
    ax.bar(x + w/2, b_ct2.mean(0), w, yerr=b_ct2.std(0), label='CT2', color='#B71C1C', alpha=0.8, capsize=3)
    ax.set_xticks(x); ax.set_xticklabels(GS, fontsize=6, rotation=45, ha='right')
    ax.set_ylabel('growth rate b̂', fontsize=7)
    ax.set_title(label, fontsize=8)
    ax.legend(fontsize=7)
    ax.axhline(0, color='k', lw=0.5)

# ── Sign agreement per pair ────────────────────────────────────────────────
nf_new  = np.array(d_new['net_flow'])
net_sym = (nf_new + nf_new.T) / 2
sp_mat  = np.sign(net_sym)
mask    = (sp_mat != 0) & (~np.eye(10, dtype=bool))

agree_old = (np.sign(A_old) == sp_mat) & mask
agree_new = (np.sign(A_new) == sp_mat) & mask
change = agree_new.astype(int) - agree_old.astype(int)  # +1 new agree, -1 lost

im2 = ax_sa.imshow(change, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
ax_sa.set_xticks(range(10)); ax_sa.set_yticks(range(10))
ax_sa.set_xticklabels(GS, fontsize=6, rotation=45, ha='right')
ax_sa.set_yticklabels(GS, fontsize=6)
ax_sa.set_title('SA change: AGORA vs L1+L2\n(green=gained, red=lost)', fontsize=8)
plt.colorbar(im2, ax=ax_sa, shrink=0.7, ticks=[-1,0,1])

fig.suptitle('AGORA W=1.0 vs L1+L2 expanded: A matrix & b̂ CT analysis', fontsize=10)
out = FIG / 'fig_agora_w1p0_analysis.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')

# ── Print b_p stats ────────────────────────────────────────────────────────
print(f'\nCT1 patients: {ct1_pat}')
print(f'CT2 patients: {ct2_pat}')
print(f'\nb̂ (AGORA W=1.0) — mean per guild:')
print(f'{"Guild":20s}  {"CT1 mean":>10s}  {"CT2 mean":>10s}  {"diff":>8s}')
for i, g in enumerate(GUILD_ORDER):
    m1 = b_new[ct1_idx, i].mean()
    m2 = b_new[ct2_idx, i].mean()
    print(f'{g:20s}  {m1:10.4f}  {m2:10.4f}  {m2-m1:+8.4f}')
