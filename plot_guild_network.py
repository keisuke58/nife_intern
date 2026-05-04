#!/usr/bin/env python3
"""Guild interaction network: A matrix as graph with KEGG sign overlay."""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

HERE = Path(__file__).parent
RES  = HERE / 'results' / 'dieckow_cr'
sys.path.insert(0, str(HERE))
from guild_replicator_dieckow import GUILD_SHORT_LIST, GUILD_COLORS_LIST, N_G
from loo_cv_kegg_prior import build_net_flow

d     = json.load(open(RES / 'fit_glv_hamilton_kegg_prior.json'))
A_opt = np.array(d['A'])

net_flow = build_net_flow()
net_sym  = (net_flow + net_flow.T) / 2.0
sp_mat   = np.sign(net_sym)
mask     = (sp_mat != 0) & (~np.eye(N_G, dtype=bool))

# ── Node positions: circular layout ──────────────────────────────────────────
angles = np.linspace(0, 2*np.pi, N_G, endpoint=False) - np.pi/2
R = 1.0
pos = np.stack([R * np.cos(angles), R * np.sin(angles)], axis=1)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect('equal')
ax.axis('off')
ax.set_xlim(-1.65, 1.65)
ax.set_ylim(-1.65, 1.65)

A_abs_max = np.abs(A_opt[~np.eye(N_G, dtype=bool)]).max()

def _edge_color(i, j):
    if not mask[i, j]:
        return '#bbbbbb', 0.45          # unconstrained → light grey
    agree = (np.sign(A_opt[i, j]) == sp_mat[i, j])
    if agree:
        return '#2ecc71', 0.85          # KEGG-consistent → green
    return '#e74c3c', 0.85              # mismatch → red

def _arrow(ax, src, tgt, color, alpha, lw, rad=0.25):
    """Curved arrow from src to tgt."""
    ax.annotate('',
        xy=tgt, xytext=src,
        arrowprops=dict(
            arrowstyle='->', color=color, alpha=alpha, lw=lw,
            connectionstyle=f'arc3,rad={rad}',
        ))

THRESH = 0.04   # skip very weak edges

# Draw undirected pairs: one arc per pair, color by KEGG agreement, style by sign
drawn = set()
for j in range(N_G):
    for i in range(N_G):
        if i == j or (i, j) in drawn or (j, i) in drawn:
            continue
        aij = A_opt[i, j]
        aji = A_opt[j, i]
        # symmetric entry — use average magnitude
        avg = (abs(aij) + abs(aji)) / 2.0
        if avg < THRESH:
            drawn.add((i, j))
            continue

        lw    = 0.6 + 5.0 * avg / A_abs_max
        color, alpha = _edge_color(i, j)

        # sign of symmetric part
        sym_val = (net_sym[i, j] + net_sym[j, i]) / 2.0 if mask[i,j] else aij
        is_pos  = (aij + aji) >= 0

        # draw each direction separately with small offset
        for src, tgt, val in [(j, i, aij), (i, j, aji)]:
            if abs(val) < THRESH:
                continue
            rad = 0.18 if val > 0 else -0.18
            lw_dir = 0.5 + 4.5 * abs(val) / A_abs_max
            col_dir, alp_dir = _edge_color(tgt, src)
            _arrow(ax, pos[src], pos[tgt], col_dir, alp_dir, lw_dir, rad=rad)

        drawn.add((i, j))

# ── Nodes ─────────────────────────────────────────────────────────────────────
NODE_R = 0.13
for k in range(N_G):
    circle = plt.Circle(pos[k], NODE_R, color=GUILD_COLORS_LIST[k],
                         zorder=5, linewidth=1.5, edgecolor='#333')
    ax.add_patch(circle)
    # label offset away from centre
    lx = pos[k, 0] * 1.28
    ly = pos[k, 1] * 1.28
    ax.text(lx, ly, GUILD_SHORT_LIST[k], ha='center', va='center',
            fontsize=9, fontweight='bold', zorder=6,
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))

# self-inhibition arcs (diagonal of A)
for k in range(N_G):
    v = A_opt[k, k]
    if v < 0:
        arc_r = NODE_R * 1.35
        theta = np.linspace(angles[k] + 0.4, angles[k] + 2*np.pi - 0.4, 80)
        xs = pos[k, 0] + arc_r * np.cos(theta)
        ys = pos[k, 1] + arc_r * np.sin(theta)
        ax.plot(xs, ys, color='#555', lw=0.8, alpha=0.5, zorder=4)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(color='#2ecc71', alpha=0.85, label='KEGG-consistent directed edge'),
    mpatches.Patch(color='#e74c3c', alpha=0.85, label='KEGG-discordant directed edge'),
    mpatches.Patch(color='#bbbbbb', alpha=0.45, label='Unconstrained directed edge'),
    mpatches.Patch(fc='none', ec='none', label='Arrow width ∝ |Aᵢⱼ|  (threshold 0.04)'),
    mpatches.Patch(fc='none', ec='none', label='Arc curves outward (+) / inward (−)'),
]
ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
          fontsize=9, frameon=True, framealpha=0.92, ncol=2)

n_agree = int(((np.sign(A_opt) == sp_mat) & mask).sum() // 2)
n_tot   = int(mask.sum() // 2)
ax.set_title(
    f'Hamilton + KEGG prior — Guild Interaction Network\n'
    f'Sign agreement: {n_agree}/{n_tot} constrained pairs  |  '
    f'Edge width ∝ |Aᵢⱼ|  |  Threshold |Aᵢⱼ| > {THRESH}',
    fontsize=12, fontweight='bold', pad=14)

for ext in ('pdf', 'png'):
    fig.savefig(RES / f'fig_guild_network.{ext}', dpi=180, bbox_inches='tight')
plt.close(fig)
print(f'Saved fig_guild_network.*')
