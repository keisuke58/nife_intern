#!/usr/bin/env python3
"""
plot_fitting_results.py
Per-patient fitting results: observed vs predicted (weeks 2 & 3)
using fit_glv_hamilton_kegg_expanded.json (Hamilton ODE, 34-pair prior, 10 patients).
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys, os
import numpy as np
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp

FIT  = _here / 'results' / 'dieckow_cr' / 'fit_glv_hamilton_kegg_expanded.json'
PHI  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
OUT  = _here / 'results' / 'dieckow_cr'

d        = json.load(open(FIT))
A_np     = np.array(d['A'])
b_all    = np.array(d['b_all'])
PATIENTS = d['patients']
GUILDS   = d['guilds']
phi_all  = np.load(PHI)
n_sp     = len(GUILDS)
NSTEPS   = 100

theta_A = jnp.array([A_np[i, j] for j in range(n_sp) for i in range(j + 1)])

@jax.jit
def run_week(b_p, phi0):
    theta = jnp.concatenate([theta_A, b_p])
    out   = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=NSTEPS, dt=1e-4,
                            phi_init=phi0, c_const=25.0, alpha_const=100.0)
    eq = out[-1]; s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

print('Computing predictions...', flush=True)
pred2, pred3 = [], []
for pi, pat in enumerate(PATIENTS):
    p2 = np.array(run_week(jnp.array(b_all[pi]), jnp.array(phi_all[pi, 0])))
    p3 = np.array(run_week(jnp.array(b_all[pi]), jnp.array(p2)))
    pred2.append(p2); pred3.append(p3)
    r2 = float(np.sqrt(np.mean((p2 - phi_all[pi,1])**2)))
    r3 = float(np.sqrt(np.mean((p3 - phi_all[pi,2])**2)))
    print(f'  {pat}: wk2={r2:.4f}  wk3={r3:.4f}', flush=True)

# ── Figure ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

COLORS = ['#4C8DC4','#5BB85B','#E0873A','#9B59B6','#D62728',
          '#17BECF','#BCBD22','#F7B6D2','#8C6D31','#AAAAAA']
SHORT  = ['Actin.','Bacil.','Bact.','β-Prot.','Clost.',
          'Coriob.','Fusob.','γ-Prot.','Negat.','Other']

plt.rcParams.update({'font.family':'serif',
                     'font.serif':['Times New Roman','Times','DejaVu Serif'],
                     'mathtext.fontset':'stix',
                     'font.size':9,
                     'axes.spines.top':False,'axes.spines.right':False})

fig, axes = plt.subplots(2, 5, figsize=(16, 7.5),
                          gridspec_kw={'hspace':0.58, 'wspace':0.13})

# x layout per patient:  W1obs | W2obs W2pred | W3obs W3pred
X  = [0.0, 1.1, 1.65, 2.75, 3.30]
W  = 0.42
XL = ['W1\nobs','W2\nobs','W2\npred','W3\nobs','W3\npred']

for pi, (pat, ax) in enumerate(zip(PATIENTS, axes.ravel())):
    obs  = phi_all[pi]          # (3, n_sp)
    p2   = pred2[pi]
    p3   = pred3[pi]

    bars = [obs[0], obs[1], p2, obs[2], p3]

    bot = np.zeros(5)
    for gi in range(n_sp):
        vals = np.array([b[gi] for b in bars])
        # solid for obs (cols 0,1,3), hatched for pred (cols 2,4)
        for bi, (xp, v, b_) in enumerate(zip(X, vals, bot)):
            hatch = '////' if bi in (2, 4) else ''
            ec    = 'white' if bi in (2, 4) else 'none'
            ax.bar(xp, v, W, bottom=b_, color=COLORS[gi],
                   hatch=hatch, edgecolor=ec, linewidth=0.35)
        bot += vals

    # separators between obs/pred pairs
    for xs in [0.87, 2.47]:
        ax.axvline(xs, color='#ccc', lw=0.8, ls='--')

    rmse_v2 = float(np.sqrt(np.mean((p2 - obs[1])**2)))
    rmse_v3 = float(np.sqrt(np.mean((p3 - obs[2])**2)))
    # Pearson r at wk2
    r2 = float(np.corrcoef(p2, obs[1])[0,1]) if p2.std() > 1e-6 else 0

    ax.set_title(f'Patient {pat}', fontsize=10, fontweight='bold', pad=3)
    ax.text(0.5, -0.27,
            f'RMSE  wk2={rmse_v2:.3f}  wk3={rmse_v3:.3f}    r={r2:.2f}',
            transform=ax.transAxes, ha='center', fontsize=7.5, color='#444')

    ax.set_xlim(-0.28, 3.60)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(X)
    ax.set_xticklabels(XL, fontsize=7.5)
    ax.set_yticks([0, 0.5, 1.0])
    if pi % 5 == 0:
        ax.set_yticklabels(['0','0.5','1'], fontsize=7.5)
        ax.set_ylabel('Guild fraction φ', fontsize=8.5)
    else:
        ax.set_yticklabels(['','',''])

# ── Global RMSE & r ───────────────────────────────────────────────────────────
all_obs2  = np.vstack([phi_all[pi,1] for pi in range(10)]).ravel()
all_obs3  = np.vstack([phi_all[pi,2] for pi in range(10)]).ravel()
all_pred2 = np.vstack(pred2).ravel()
all_pred3 = np.vstack(pred3).ravel()
rmse_all  = float(np.sqrt((np.mean((all_pred2-all_obs2)**2) +
                            np.mean((all_pred3-all_obs3)**2)) / 2))
r_all     = float(np.corrcoef(np.concatenate([all_pred2,all_pred3]),
                               np.concatenate([all_obs2, all_obs3]))[0,1])

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=COLORS[i], label=SHORT[i]) for i in range(n_sp)]
patches += [
    mpatches.Patch(facecolor='#aaa', edgecolor='none',    label='Observed (solid)'),
    mpatches.Patch(facecolor='#aaa', hatch='////', edgecolor='white', label='Predicted (hatched)'),
]
fig.legend(handles=patches, loc='lower center', ncol=12,
           fontsize=8, frameon=True, framealpha=0.92,
           bbox_to_anchor=(0.5, -0.03),
           title='Guild  |  pattern', title_fontsize=8)

fig.suptitle(
    f'Fitting Results — Hamilton ODE + Expanded Dieckow+AGORA Prior (34 pairs, 10 patients)\n'
    f'Training RMSE = {rmse_all:.4f}  ·  Pearson r = {r_all:.3f}  ·  SA = 64/68 (94%)',
    fontsize=11, fontweight='bold', y=1.02)

for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'fig_fitting_results.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'\nOverall: RMSE={rmse_all:.4f}  r={r_all:.3f}')
print('Saved fig_fitting_results.*')
