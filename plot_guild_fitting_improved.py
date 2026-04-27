#!/usr/bin/env python3
"""
plot_guild_fitting_improved.py — Improved guild-level gLV fitting visualization.

Two outputs:
  fig12b_guild_fitting_stacked.pdf  — 2×4 per-patient stacked bars (W1|W2obs|W2pred|W3obs|W3pred)
  fig11b_guild_scatter.pdf          — predicted vs observed scatter (r=0.963 highlight)

Run from nife/ directory:
  python plot_guild_fitting_improved.py
"""

import json, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
import pub_style; pub_style.apply()
from guild_replicator_dieckow import (
    GUILD_ORDER, GUILD_SHORT, GUILD_COLORS, GUILD_COLORS_LIST,
    predict_trajectory,
)

CR_DIR  = _here / 'results' / 'dieckow_cr'
OTU_DIR = _here / 'results' / 'dieckow_otu'
DOCS    = Path(__file__).resolve().parents[1] / 'docs' / 'figures' / 'dieckow'

FIT_JSON = CR_DIR / 'fit_guild.json'
PHI_NPY  = OTU_DIR / 'phi_guild.npy'

# ── Load data ─────────────────────────────────────────────────────────────────
fit   = json.load(FIT_JSON.open())
A     = np.array(fit['A'])
b_all = np.array(fit['b_all'])
PATIENTS = fit['patients']          # 10 patients (A-L excl I,J)
N_P  = len(PATIENTS)
N_G  = A.shape[0]

phi_10 = np.load(PHI_NPY)          # (10, 3, 10)
PAT10  = list('ABCDEFGHKL')
pat_idx = [PAT10.index(p) for p in PATIENTS]
phi_all = phi_10[pat_idx][:, :, :N_G]  # (N_P, 3, N_G)

# Guilds and colours
guilds = fit.get('guilds', GUILD_ORDER[:N_G])
SHORT  = [GUILD_SHORT.get(g, g[:6]) for g in guilds]
GCOLS  = [GUILD_COLORS.get(g, '#888888') for g in guilds]


def replicator_step(phi0, b, A):
    def rhs(t, phi):
        f = b + A @ phi; fbar = phi @ f
        return phi * (f - fbar)
    sol = solve_ivp(rhs, [0, 1.0], phi0, method='RK45', rtol=1e-6, atol=1e-8)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0


def compute_predictions():
    pred2 = np.zeros((N_P, N_G))
    pred3 = np.zeros((N_P, N_G))
    for i in range(N_P):
        pred2[i] = replicator_step(phi_all[i, 0], b_all[i], A)
        pred3[i] = replicator_step(pred2[i], b_all[i], A)
    return pred2, pred3


print('Computing predictions...', flush=True)
pred_W2, pred_W3 = compute_predictions()

rmse_list = [
    float(np.sqrt(np.mean(
        (np.stack([pred_W2[i], pred_W3[i]]) - phi_all[i, 1:]) ** 2
    )))
    for i in range(N_P)
]


# ══════════════════════════════════════════════════════════════════════════════
# Fig 12b — Per-patient stacked bars  (2×4 grid)
# ══════════════════════════════════════════════════════════════════════════════
print('Generating fig12b...', flush=True)

ncols = 5; nrows = (N_P + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.5),
                         gridspec_kw={'hspace': 0.50, 'wspace': 0.12})
axes = axes.flatten()

bw   = 0.70      # bar width
# x positions: W1-obs | [gap] | W2-obs, W2-pred | [gap] | W3-obs, W3-pred
x_w1   = 0.0
x_w2_o = 1.55;  x_w2_p = x_w2_o + bw + 0.06
x_w3_o = 3.30;  x_w3_p = x_w3_o + bw + 0.06

xs_obs  = [x_w1, x_w2_o, x_w3_o]    # W1, W2, W3 observed
xs_pred = [x_w2_p, x_w3_p]           # W2, W3 predicted

# Colour + alpha per bar type: obs=full, pred=lighter+hatch
for pi, pat in enumerate(PATIENTS):
    ax   = axes[pi]
    obs  = phi_all[pi]     # (3, N_G)
    pred = [pred_W2[pi], pred_W3[pi]]

    # Observed bars (W1, W2, W3) — solid
    for wi, x0 in enumerate(xs_obs):
        bottom = 0.0
        for g in range(N_G):
            ax.bar(x0, obs[wi, g], bw, bottom=bottom,
                   color=GCOLS[g], alpha=0.90, linewidth=0)
            bottom += obs[wi, g]

    # Predicted bars (W2, W3) — hatched, lighter
    for wi, x0 in enumerate(xs_pred):
        bottom = 0.0
        for g in range(N_G):
            ax.bar(x0, pred[wi][g], bw, bottom=bottom,
                   color=GCOLS[g], alpha=0.40,
                   edgecolor=GCOLS[g], linewidth=0.5,
                   hatch='////')
            bottom += pred[wi][g]

    # Week group labels on x-axis
    tick_x = [x_w1, (x_w2_o + x_w2_p) / 2, (x_w3_o + x_w3_p) / 2]
    ax.set_xticks(tick_x)
    ax.set_xticklabels(['W1', 'W2', 'W3'], fontsize=10)
    ax.set_xlim(-0.55, x_w3_p + 0.55)
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.5, 1.0])

    # Obs | Pred sub-labels
    for x0, lbl in [(x_w2_o, 'obs'), (x_w2_p, 'pred'),
                    (x_w3_o, 'obs'), (x_w3_p, 'pred')]:
        ax.text(x0, -0.09, lbl, ha='center', va='top',
                fontsize=7.0, color='#555555',
                transform=ax.get_xaxis_transform())

    if pi % ncols == 0:
        ax.set_ylabel('Relative abundance', fontsize=9)
    else:
        ax.set_yticklabels([])

    # Dividers between week groups
    for xd in [1.25, 3.00]:
        ax.axvline(xd, color='#cccccc', lw=0.7, ls='--', zorder=0)

    # RMSE badge
    ax.text(0.97, 0.97, f'RMSE = {rmse_list[pi]:.3f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.25', fc='white',
                      ec='#cccccc', lw=0.6, alpha=0.85))
    ax.set_title(f'Patient {pat}', fontsize=11, fontweight='bold', pad=5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# ── Legend ────────────────────────────────────────────────────────────────────
sp_patches = [Patch(color=c, alpha=0.90, label=s)
              for c, s in zip(GCOLS, SHORT)]
obs_patch  = Patch(color='#888888', alpha=0.90, label='Observed (solid)')
pred_patch = Patch(color='#888888', alpha=0.40, label='Predicted MAP (hatched)',
                   hatch='////')
fig.legend(handles=sp_patches + [obs_patch, pred_patch],
           loc='lower center', ncol=7, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.02), frameon=False, handlelength=1.5)

r_val_train = 0.963
fig.suptitle(
    f'Guild-level gLV fit — Dieckow {N_P}-patient cohort  '
    f'(RMSE = {fit["rmse"]:.4f},  r = {r_val_train})\n'
    'Solid bars = observed; hatched bars = MAP prediction (W2, W3)',
    fontsize=11, fontweight='bold', y=1.01)

for ext in ('pdf', 'png'):
    fig.savefig(CR_DIR / f'fig12b_guild_fitting_stacked.{ext}',
                dpi=300, bbox_inches='tight')
    fig.savefig(DOCS / f'fig12b_guild_fitting_stacked.{ext}',
                dpi=300, bbox_inches='tight')
plt.close()
print('  Saved fig12b_guild_fitting_stacked', flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fig 11b — Predicted vs observed scatter  (improved version)
# ══════════════════════════════════════════════════════════════════════════════
print('Generating fig11b...', flush=True)

# Pool W2 and W3 observations/predictions
obs_all, pred_all, col_all = [], [], []
for i in range(N_P):
    for g in range(N_G):
        obs_all.append(phi_all[i, 1, g])
        pred_all.append(pred_W2[i, g])
        col_all.append(GCOLS[g])
        obs_all.append(phi_all[i, 2, g])
        pred_all.append(pred_W3[i, g])
        col_all.append(GCOLS[g])

obs_all  = np.array(obs_all)
pred_all = np.array(pred_all)
r_val = np.corrcoef(obs_all, pred_all)[0, 1]

fig2, ax2 = plt.subplots(figsize=(5.5, 5.2))
for g in range(N_G):
    mask = np.array([GCOLS[g]] * len(obs_all)) == np.array(col_all)
    ax2.scatter(obs_all[mask], pred_all[mask],
                color=GCOLS[g], s=40, alpha=0.80,
                linewidths=0.4, edgecolors='white', zorder=3,
                label=SHORT[g])

lim = 1.02
ax2.plot([0, lim], [0, lim], 'k--', lw=1.0, alpha=0.5, label='y = x')
ax2.set_xlim(-0.01, lim); ax2.set_ylim(-0.01, lim)
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_xlabel('Observed relative abundance', fontsize=12)
ax2.set_ylabel('Predicted relative abundance', fontsize=12)
ax2.set_aspect('equal')
ax2.set_title('Guild-level gLV: Predicted vs. Observed\n'
              '(all patients × W2 + W3, 10 guilds)',
              fontsize=12, fontweight='bold')
ax2.text(0.04, 0.96,
         f'RMSE = {fit["rmse"]:.4f}\n$r$ = {r_val:.3f}\n'
         f'N = {N_P} patients × 2 weeks × {N_G} guilds',
         transform=ax2.transAxes, fontsize=10, va='top',
         bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#cccccc',
                   lw=0.8, alpha=0.9))
ax2.legend(ncol=2, fontsize=8.5, frameon=False,
           loc='lower right', handletextpad=0.4)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
plt.tight_layout()

for ext in ('pdf', 'png'):
    fig2.savefig(CR_DIR / f'fig11b_guild_scatter.{ext}',
                 dpi=300, bbox_inches='tight')
    fig2.savefig(DOCS / f'fig11b_guild_scatter.{ext}',
                 dpi=300, bbox_inches='tight')
plt.close()
print('  Saved fig11b_guild_scatter', flush=True)
print('Done.')
