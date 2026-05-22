#!/usr/bin/env python3
"""Compact trajectory figure for slide: scatter plot + 3 representative patients.
Uses Hamilton ODE (simulate_0d_nsp) for correct predictions."""
import json, sys, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

HERE = Path(__file__).parent
RES  = HERE / 'results' / 'dieckow_cr'

from guild_replicator_dieckow import GUILD_SHORT_LIST, GUILD_COLORS_LIST, N_G
from loo_cv_kegg_prior import PHI_NPY, PATIENTS

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

# ── Load fit result ───────────────────────────────────────────────────────────
d     = json.load(open(RES / 'fit_glv_hamilton_kegg_prior.json'))
A_opt = np.array(d['A'])
b_opt = np.array(d['b_all'])
rmse  = float(d['rmse'])
r_fit = float(d['r'])

phi_all = np.load(PHI_NPY)
phi_sub = np.stack([phi_all[k] for k in range(phi_all.shape[0])
                    if phi_all[k].sum() > 1e-9])[:len(PATIENTS)]

n_sp = N_G

# ── Hamilton ODE prediction ───────────────────────────────────────────────────
sys.path.insert(0, str(HERE.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp

NSTEPS = 100

# pack A_opt → theta_A (upper-triangular, col-major)
theta_A = jnp.array([A_opt[i, j] for j in range(n_sp) for i in range(j + 1)])

@jax.jit
def predict_one(b_p, phi0):
    theta = jnp.concatenate([theta_A, b_p])
    phibar = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=NSTEPS, dt=1e-4,
                              phi_init=phi0, c_const=25.0, alpha_const=100.0)
    eq = phibar[-1]; s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

print('Compiling Hamilton predict...', flush=True)
# JIT warmup
_ = predict_one(jnp.array(b_opt[0]), jnp.array(phi_sub[0, 0]))
print('Done.', flush=True)

# ── Collect predictions ───────────────────────────────────────────────────────
all_obs, all_pred = [], []
per_pat = {}
for pi, pat in enumerate(PATIENTS):
    phi_obs = phi_sub[pi]
    b_p     = jnp.array(b_opt[pi])
    p2 = np.array(predict_one(b_p, jnp.array(phi_obs[0])))
    p3 = np.array(predict_one(b_p, jnp.array(p2)))
    per_pat[pat] = (phi_obs, p2, p3)
    all_obs  += phi_obs[1].tolist() + phi_obs[2].tolist()
    all_pred += p2.tolist()          + p3.tolist()
    r_p = float(np.corrcoef(
        phi_obs[1].tolist() + phi_obs[2].tolist(), p2.tolist() + p3.tolist())[0, 1])
    rmse_p = float(np.sqrt(np.mean((p2 - phi_obs[1])**2 + (p3 - phi_obs[2])**2) / 2))
    print(f'  {pat}: RMSE={rmse_p:.4f}  r={r_p:.3f}')

all_obs  = np.array(all_obs)
all_pred = np.array(all_pred)
r_all    = float(np.corrcoef(all_obs, all_pred)[0, 1])
print(f'Overall r={r_all:.3f}  RMSE={float(np.sqrt(np.mean((all_obs-all_pred)**2))):.4f}')

# ── Figure ────────────────────────────────────────────────────────────────────
# representative: best G (RMSE=0.025), typical H (RMSE=0.090), worst K (RMSE=0.163)
SHOW = [('G', 'Best — Patient G'),
        ('H', 'Typical — Patient H'),
        ('K', 'Worst — Patient K')]
GUILD_SHORT = GUILD_SHORT_LIST

fig = plt.figure(figsize=(13, 5.2))
gs  = GridSpec(3, 3, figure=fig,
               left=0.07, right=0.97, top=0.90, bottom=0.13,
               hspace=0.72, wspace=0.35,
               width_ratios=[1.15, 1, 1])

# ── Scatter ───────────────────────────────────────────────────────────────────
ax_sc = fig.add_subplot(gs[:, 0])
guild_cols = np.tile(GUILD_COLORS_LIST, len(PATIENTS) * 2)
ax_sc.scatter(all_obs, all_pred, c=guild_cols[:len(all_obs)],
              s=22, alpha=0.72, edgecolors='none', zorder=3)

lim = max(all_obs.max(), all_pred.max()) * 1.06
ax_sc.plot([0, lim], [0, lim], 'k--', lw=1.0, alpha=0.5)
ax_sc.fill_between([0, lim], [rmse, lim + rmse], [-rmse, lim - rmse],
                   alpha=0.10, color='steelblue')
ax_sc.set_xlim(-0.01, lim); ax_sc.set_ylim(-0.01, lim)
ax_sc.set_xlabel('Observed fraction')
ax_sc.set_ylabel('Predicted fraction')
ax_sc.set_title('All patients & guilds\n(weeks 2 & 3)', fontsize=10.5, fontweight='bold')
ax_sc.text(0.04, 0.96, f'r = {r_all:.3f}\nRMSE = {rmse:.4f}',
           transform=ax_sc.transAxes, va='top', fontsize=9.5,
           bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#ccc', alpha=0.9))

# ── Bar panels ────────────────────────────────────────────────────────────────
x = np.arange(N_G); w = 0.38

for row, (pat, label) in enumerate(SHOW):
    phi_obs, p2, p3 = per_pat[pat]
    for col, (phi_pred, phi_true, wk) in enumerate([
            (p2, phi_obs[1], 'Wk 2'), (p3, phi_obs[2], 'Wk 3')]):
        ax = fig.add_subplot(gs[row, col + 1])
        rmse_w = float(np.sqrt(np.mean((phi_pred - phi_true)**2)))
        r_w    = float(np.corrcoef(phi_true, phi_pred)[0, 1])

        data_max = max(phi_true.max(), phi_pred.max())
        ymax     = min(1.0, data_max * 1.4 + 0.03)
        FLOOR    = ymax * 0.04   # 4% of y range — always visible

        obs_plot  = np.where(phi_true > 0, np.maximum(phi_true, FLOOR), 0.0)
        pred_plot = np.where(phi_pred > 0, np.maximum(phi_pred, FLOOR), 0.0)

        ax.bar(x - w/2, obs_plot,  w, color=GUILD_COLORS_LIST,
               alpha=0.9,  edgecolor='k', linewidth=0.35)
        ax.bar(x + w/2, pred_plot, w, color=GUILD_COLORS_LIST,
               alpha=0.5,  edgecolor='k', linewidth=0.35, hatch='//')

        # mark floor-clipped bars with a subtle marker
        for gi in range(N_G):
            if 0 < phi_true[gi] < FLOOR:
                ax.text(x[gi]-w/2, FLOOR*1.05, '▲', ha='center', va='bottom',
                        fontsize=4, color='#777')
            if 0 < phi_pred[gi] < FLOOR:
                ax.text(x[gi]+w/2, FLOOR*1.05, '▲', ha='center', va='bottom',
                        fontsize=4, color='#777')
        ax.set_ylim(0, min(1.0, ymax))
        ax.set_xlim(-0.6, N_G - 0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(GUILD_SHORT, fontsize=5.5, rotation=38, ha='right')
        ax.tick_params(axis='y', labelsize=7)

        row_label = f'{label}\n' if col == 0 else ''
        ax.set_title(f'{row_label}{wk}  RMSE={rmse_w:.3f}', fontsize=8.5, pad=3)
        ax.text(0.97, 0.96, f'r={r_w:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=7.5, color='#555')
        if col == 0:
            ax.set_ylabel('Fraction', fontsize=8)

# column headers
for ci, wk_label in enumerate(['Week 2', 'Week 3']):
    fig.text(0.395 + ci * 0.295, 0.935, wk_label,
             ha='center', fontsize=10, fontweight='bold', color='#1a3a5c')

# legend
from matplotlib.patches import Patch
handles = [
    Patch(fc='#888', ec='k', lw=0.4, alpha=0.9,  label='Observed'),
    Patch(fc='#888', ec='k', lw=0.4, alpha=0.5,  hatch='//', label='Predicted'),
]
fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.985, 0.97),
           fontsize=9, frameon=True, framealpha=0.9)

fig.suptitle(
    f'Hamilton + KEGG prior — Predicted vs Observed  '
    f'(r = {r_all:.3f},  RMSE = {rmse:.4f},  Sign agreement = 22/22)',
    fontsize=11.5, fontweight='bold', y=0.98)

for ext in ('pdf', 'png'):
    fig.savefig(RES / f'fig_slide_trajectory.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Saved fig_slide_trajectory.*')
