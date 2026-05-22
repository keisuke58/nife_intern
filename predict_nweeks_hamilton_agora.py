#!/usr/bin/env python3
"""
predict_nweeks_hamilton_agora.py
Extrapolate the 10-guild Hamilton ODE (AGORA W=1.0) beyond training window.
MAP A + per-patient b̂ → predict weeks 2, 3 (validation) and 4, 5, 10 (extrapolation).
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from guild_replicator_dieckow import GUILD_ORDER, N_G

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp

CR  = _here / 'results' / 'dieckow_cr'
OTU = _here / 'results' / 'dieckow_otu'
FIG = CR / 'figs'
FIG.mkdir(exist_ok=True)

# Load fit
d = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))
A_opt  = jnp.array(d['A'])
b_all  = jnp.array(d['b_all'])   # (10, 10)
PATIENTS = d['patients']

GS = [g[:6] for g in GUILD_ORDER]

# Load observed data
phi_all = np.load(OTU / 'phi_guild.npy')   # (10, 3, 10)
ct_map  = json.load(open(OTU / 'community_types.json'))['patient_ct']

NSTEPS = 100; DT = 1e-4; C = 25.0; ALPHA = 100.0
n_sp = N_G

theta_A = jnp.array([A_opt[i, j] for j in range(n_sp) for i in range(j + 1)])

def _run_sim(theta, phi_init):
    phibar = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=NSTEPS, dt=DT,
                              phi_init=phi_init, c_const=C, alpha_const=ALPHA)
    eq = phibar[-1]; s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

run_sim_j = jax.jit(_run_sim)

MAX_WEEK = 10
N_WEEKS_PRED = MAX_WEEK  # predict week 1 through MAX_WEEK

# Predict trajectories
preds = {}  # patient → (MAX_WEEK, n_sp) array
for pi, pat in enumerate(PATIENTS):
    theta = jnp.concatenate([theta_A, b_all[pi]])
    phi_t = [np.array(phi_all[pi, 0])]  # week 1 = observed
    phi_cur = jnp.array(phi_all[pi, 0])
    for _ in range(MAX_WEEK - 1):
        phi_cur = run_sim_j(theta, phi_cur)
        phi_t.append(np.array(phi_cur))
    preds[pat] = np.array(phi_t)  # (MAX_WEEK, n_sp)

# RMSE at each week (weeks 2 and 3 have observed data)
print('\nWeek-by-week RMSE (weeks 2 & 3 are observed, 4-10 are extrapolations):')
for wk in range(2, 11):
    if wk <= 3:
        obs_idx = wk - 1  # phi_all[:,0]=wk1, [:,1]=wk2, [:,2]=wk3
        rmses = []
        for pi, pat in enumerate(PATIENTS):
            pred_wk = preds[pat][wk - 1]
            obs_wk  = phi_all[pi, obs_idx]
            rmses.append(np.sqrt(np.mean((pred_wk - obs_wk) ** 2)))
        tag = '(observed)' if wk <= 3 else '(extrap.)'
        print(f'  Week {wk}: mean RMSE = {np.mean(rmses):.4f} ± {np.std(rmses):.4f} {tag}')

# ── Figure: trajectories ──────────────────────────────────────────────────────
weeks = np.arange(1, MAX_WEEK + 1)
n_pat = len(PATIENTS)
n_rows = (n_pat + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(16, n_rows * 3.5))
plt.rcParams.update({'font.size': 9, 'font.family':'serif',
                     'font.serif':['Times New Roman','Times','DejaVu Serif'],
                     'mathtext.fontset':'stix'})

GUILD_COLORS = {g: c for g, c in zip(GUILD_ORDER, [
    '#976832', '#20af53', '#ea2323', '#4090d0', '#98c557',
    '#b69e7c', '#ffa500', '#707070', '#703a98', '#444444'
])}

axes_flat = axes.flatten()
for pi, pat in enumerate(PATIENTS):
    ax = axes_flat[pi]
    phi_pred = preds[pat]   # (MAX_WEEK, n_sp)
    for gi, guild in enumerate(GUILD_ORDER):
        c = GUILD_COLORS[guild]
        ax.plot(weeks, phi_pred[:, gi], color=c, lw=1.5, label=GS[gi])
        # observed weeks 1-3 as dots
        for wk in range(1, 4):
            ax.scatter(wk, phi_all[pi, wk - 1, gi], color=c, s=30, zorder=5)

    ax.axvspan(3.5, MAX_WEEK, alpha=0.07, color='gray', label='extrapolation')
    ax.axvline(3.5, color='gray', ls=':', lw=1)
    ct = ct_map[pat]
    ax.set_title(f'Patient {pat} (CT{ct})', fontsize=10)
    ax.set_xlim(0.5, MAX_WEEK + 0.5)
    ax.set_ylim(-0.01, 1.0)
    ax.set_xlabel('Week')
    ax.set_ylabel('Fraction')

# hide unused axes
for ax in axes_flat[n_pat:]:
    ax.set_visible(False)

# legend (use last panel)
handles, labels = axes_flat[0].get_legend_handles_labels()
legend_handles = handles[:-1]  # exclude the extrapolation patch for guild legend
axes_flat[1].legend(handles[:n_sp], labels[:n_sp], fontsize=7, loc='upper right',
                    ncol=2, framealpha=0.7)

fig.suptitle('AGORA W=1.0 Hamilton ODE: 10-week extrapolation\n'
             'Dots = observed (wk1–3); lines = predicted; shaded = extrapolation region',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = FIG / 'fig_agora_w1p0_nweeks_extrapolation.png'
fig.savefig(out, dpi=300, bbox_inches='tight')
fig.savefig(FIG / 'fig_agora_w1p0_nweeks_extrapolation.pdf', bbox_inches='tight')
print(f'Saved {out} + PDF')
