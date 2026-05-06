#!/usr/bin/env python3
"""
compare_all_models_bc.py
  Compare RMSE and Bray-Curtis across all models using the correct ODE evaluator.

  Evaluator rule:
  - gLV free (trained with guild_replicator replicator ODE): predict_trajectory()
  - Hamilton models (trained with hamilton_ode_jax_nsp): simulate_0d_nsp()
"""
import json, sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

os.environ['CUDA_VISIBLE_DEVICES'] = ''

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
from guild_replicator_dieckow import GUILD_ORDER, N_G, predict_trajectory

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp

CR  = _here / 'results' / 'dieckow_cr'
OTU = _here / 'results' / 'dieckow_otu'
FIG = CR / 'figs'
FIG.mkdir(exist_ok=True)

PATIENTS = list('ABCDEFGHKL')
phi_all = np.load(OTU / 'phi_guild.npy')   # (10, 3, 10)

NSTEPS   = 100
DT       = 1e-4
C_CONST  = 25.0
ALPHA    = 100.0

# ── Bray-Curtis ──────────────────────────────────────────────────────────────
def bray_curtis(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return 1.0 - 2.0 * np.minimum(x, y).sum() / (x.sum() + y.sum())

# ── Pack symmetric A → upper-triangular theta_A (column-major) ──────────────
def pack_theta_A(A_mat):
    n = A_mat.shape[0]
    return jnp.array([A_mat[i, j] for j in range(n) for i in range(j + 1)])

# ── Hamilton NSP evaluator ───────────────────────────────────────────────────
def _nsp_step(theta, phi0):
    eq = simulate_0d_nsp(theta, n_sp=N_G, n_steps=NSTEPS, dt=DT,
                          phi_init=phi0, c_const=C_CONST, alpha_const=ALPHA)[-1]
    s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(N_G) / N_G)

_nsp_jit = jax.jit(_nsp_step)

def evaluate_hamilton(fit_dict, phi_obs):
    A_mat  = np.array(fit_dict['A'])
    b_all  = np.array(fit_dict['b_all'])
    theta_A = pack_theta_A(A_mat)
    n_p = phi_obs.shape[0]
    rows = []
    for p in range(n_p):
        theta = jnp.concatenate([theta_A, jnp.array(b_all[p])])
        p2 = np.array(_nsp_jit(theta, jnp.array(phi_obs[p, 0])))
        p3 = np.array(_nsp_jit(theta, jnp.array(p2)))
        for wk, pred, obs in [(2, p2, phi_obs[p, 1]), (3, p3, phi_obs[p, 2])]:
            rows.append({'patient': PATIENTS[p], 'week': wk,
                         'rmse': float(np.sqrt(np.mean((pred - obs)**2))),
                         'bc':   float(bray_curtis(pred, obs))})
    df = pd.DataFrame(rows)
    # mean RMSE = sqrt(mean squared errors over all predictions × guilds)
    rmse = float(np.sqrt(np.mean((df.rmse**2).values * N_G) / N_G))
    return df.rmse.mean(), df.bc.mean(), df

def evaluate_glv(fit_dict, phi_obs):
    A     = np.array(fit_dict['A'])
    b_all = np.array(fit_dict['b_all'])
    n_p = phi_obs.shape[0]
    rows = []
    for p in range(n_p):
        p2, p3 = predict_trajectory(phi_obs[p, 0], b_all[p], A)
        for wk, pred, obs in [(2, p2, phi_obs[p, 1]), (3, p3, phi_obs[p, 2])]:
            rows.append({'patient': PATIENTS[p], 'week': wk,
                         'rmse': float(np.sqrt(np.mean((pred - obs)**2))),
                         'bc':   float(bray_curtis(pred, obs))})
    df = pd.DataFrame(rows)
    return df.rmse.mean(), df.bc.mean(), df

# ── Model registry ────────────────────────────────────────────────────────────
# (label, fname, evaluator)
MODELS = [
    ('gLV free',                    'fit_guild_jaxlbfgs_best.json',                            'glv'),
    ('Hamilton (no prior)',          'fit_guild_hamilton.json',                                  'hamilton'),
    ('Hamilton L1+L2',               'fit_glv_hamilton_kegg_expanded.json',                     'hamilton'),
    ('Hamilton L1+L2+L3 W=0.5',     'fit_glv_hamilton_kegg_expanded_agora_w0p5.json',           'hamilton'),
    ('Hamilton L1+L2+L3 W=1.0 ★',   'fit_glv_hamilton_kegg_expanded_agora_w1p0.json',           'hamilton'),
    ('Hamilton L1+L2+L3 W=1.5',     'fit_glv_hamilton_kegg_expanded_agora_w1p5.json',           'hamilton'),
    ('Hamilton L1+L2+L3 W=2.0',     'fit_glv_hamilton_kegg_expanded_agora_w2p0.json',           'hamilton'),
    ('MacArthur c=0.01',             'fit_glv_hamilton_kegg_expanded_mac_c0p010.json',           'hamilton'),
    ('MacArthur c=0.05',             'fit_glv_hamilton_kegg_expanded_mac_c0p050.json',           'hamilton'),
]

print(f'\n{"Model":<42} {"RMSE":>8} {"BC":>8}  (stored RMSE)')
print('-' * 72)
summary = []
for label, fname, ev in MODELS:
    path = CR / fname
    if not path.exists():
        print(f'  {label:<40} — file not found')
        continue
    d = json.load(open(path))
    if ev == 'glv':
        rmse, bc, df_p = evaluate_glv(d, phi_all)
    else:
        rmse, bc, df_p = evaluate_hamilton(d, phi_all)
    stored = d.get('rmse', float('nan'))
    print(f'  {label:<40} {rmse:8.4f} {bc:8.4f}  ({stored:.4f})')
    summary.append({'model': label, 'rmse': rmse, 'bc': bc,
                    'stored_rmse': stored, 'evaluator': ev})

df_sum = pd.DataFrame(summary)
df_sum.to_csv(CR / 'all_models_bc.csv', index=False)
print(f'\nSaved {CR}/all_models_bc.csv')

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
plt.rcParams.update({'font.size': 9})

ax = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, len(df_sum)))
for i, row in df_sum.iterrows():
    mk = '*' if '★' in row.model else ('D' if row.evaluator == 'glv' else 'o')
    ax.scatter(row.rmse, row.bc, color=colors[i], marker=mk, s=80, zorder=3,
               label=row.model)
    short = row.model.split(' ')[0] if len(row.model.split(' ')[0]) > 2 else row.model[:10]
    ax.annotate(short, (row.rmse, row.bc), xytext=(3, 2),
                textcoords='offset points', fontsize=6)
ax.set_xlabel('RMSE', fontsize=9)
ax.set_ylabel('Bray-Curtis', fontsize=9)
ax.set_title('All models: RMSE vs Bray-Curtis\n(correct evaluator per model)', fontsize=9)
r, p = stats.pearsonr(df_sum.rmse, df_sum.bc)
ax.text(0.05, 0.97, f'r={r:.3f}  p={p:.2e}', transform=ax.transAxes,
        va='top', fontsize=8, bbox=dict(boxstyle='round', fc='white', alpha=0.8))

ax2 = axes[1]
order = df_sum.sort_values('bc')
bar_colors = ['#e74c3c' if '★' in m else
              '#f39c12' if 'L1+L2' == m.strip() else
              '#95a5a6' if 'gLV' in m else '#2980b9'
              for m in order.model]
ax2.barh(range(len(order)), order.bc, color=bar_colors, alpha=0.8)
ax2.set_yticks(range(len(order)))
ax2.set_yticklabels([m[:38] for m in order.model], fontsize=7.5)
ax2.set_xlabel('Bray-Curtis dissimilarity', fontsize=9)
ax2.set_title('Models ranked by BC\n(lower = better compositional accuracy)', fontsize=9)
for i, (_, row) in enumerate(order.iterrows()):
    ax2.text(row.bc + 0.002, i, f'{row.bc:.3f}', va='center', fontsize=7)

fig.suptitle('Model comparison: training RMSE and Bray-Curtis dissimilarity', fontsize=10)
fig.tight_layout()
out = FIG / 'fig_all_models_bc.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')
