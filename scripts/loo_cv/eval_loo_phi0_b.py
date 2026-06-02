#!/usr/bin/env python3
"""
eval_loo_phi0_b.py — compare two b_ho estimation strategies in LOO-CV:

  Strategy A (current): fit b_ho by minimizing RMSE on held-out wk2/wk3 [data leakage]
  Strategy B (proposed): predict b_ho from held-out phi_0 via ridge regression
                         trained on 9 training patients' (phi_0, b) pairs [truly OOS]

Usage:
    /home/nishioka/IKM_Hiwi/.venv_jax/bin/python eval_loo_phi0_b.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
from hamilton_ode_jax_nsp import simulate_0d_nsp
from guild_replicator_dieckow import GUILD_ORDER, N_G

CR       = HERE / 'results' / 'dieckow_cr'
OTU      = HERE / 'results' / 'dieckow_otu'
PATIENTS = list('ABCDEFGHKL')

phi_all = np.load(OTU / 'phi_guild.npy')   # (10, 3, 10)

# Full-fit b values — used as proxy for training patients' b in each fold
_full = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded.json'))
b_all = np.array(_full['b_all'])   # (10, 10)


def bray_curtis(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return 1.0 - 2.0 * np.minimum(x, y).sum() / (x.sum() + y.sum())


def pack_theta_A(A_mat):
    n = len(A_mat)
    return jnp.array([A_mat[i][j] for j in range(n) for i in range(j + 1)])


def predict_trajectory(theta_A, b_p, phi0):
    theta = jnp.concatenate([theta_A, jnp.array(b_p)])

    @jax.jit
    def step(phi):
        eq = simulate_0d_nsp(theta, n_sp=N_G, n_steps=100, dt=1e-4,
                             phi_init=phi, c_const=25.0, alpha_const=100.0)[-1]
        s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(N_G) / N_G)

    p2 = np.array(step(jnp.array(phi0)))
    p3 = np.array(step(jnp.array(p2)))
    return p2, p3


def estimate_b_ridge(fold, phi0_ho):
    """Predict b_ho from phi0 using RidgeCV trained on 9 training patients."""
    tr_idx = [i for i in range(10) if i != fold]
    X_tr = phi_all[tr_idx, 0]   # (9, 10) — training phi_0
    Y_tr = b_all[tr_idx]        # (9, 10) — training b

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    x_ho_s = scaler.transform(phi0_ho.reshape(1, -1))

    # RidgeCV: try alphas over 4 orders of magnitude
    alphas = np.logspace(-3, 3, 30)
    b_pred = np.zeros(N_G)
    for k in range(N_G):
        reg = RidgeCV(alphas=alphas, cv=min(5, len(tr_idx))).fit(X_tr_s, Y_tr[:, k])
        b_pred[k] = reg.predict(x_ho_s)[0]
    return b_pred


# ── Evaluate both strategies for Hamilton no-prior model ─────────────────────
print('Evaluating b_ho strategies...', flush=True)
rows = []

for fold in range(10):
    path = CR / f'loo_expanded_a0p0_fold{fold}.json'
    if not path.exists():
        print(f'  fold {fold}: missing', flush=True)
        continue

    d = json.load(open(path))
    pat      = d['patient']
    p_idx    = PATIENTS.index(pat)
    theta_A  = pack_theta_A(np.array(d['A']))
    phi0     = phi_all[p_idx, 0]
    obs2     = phi_all[p_idx, 1]
    obs3     = phi_all[p_idx, 2]

    # Strategy A: data-fitted b_ho (current)
    b_A = np.array(d['b_ho'])
    p2_A, p3_A = predict_trajectory(theta_A, b_A, phi0)
    bc_A  = (bray_curtis(p2_A, obs2) + bray_curtis(p3_A, obs3)) / 2
    rmse_A = float(np.sqrt((np.mean((p2_A-obs2)**2)+np.mean((p3_A-obs3)**2))/2))

    # Strategy B: phi0-regression b_ho
    b_B = estimate_b_ridge(fold, phi0)
    p2_B, p3_B = predict_trajectory(theta_A, b_B, phi0)
    bc_B  = (bray_curtis(p2_B, obs2) + bray_curtis(p3_B, obs3)) / 2
    rmse_B = float(np.sqrt((np.mean((p2_B-obs2)**2)+np.mean((p3_B-obs3)**2))/2))

    # Strategy C: mean b of training patients (simplest baseline)
    tr_idx = [i for i in range(10) if i != fold]
    b_C = b_all[tr_idx].mean(axis=0)
    p2_C, p3_C = predict_trajectory(theta_A, b_C, phi0)
    bc_C  = (bray_curtis(p2_C, obs2) + bray_curtis(p3_C, obs3)) / 2
    rmse_C = float(np.sqrt((np.mean((p2_C-obs2)**2)+np.mean((p3_C-obs3)**2))/2))

    print(f'  {pat}: A(data-fit) BC={bc_A:.4f}  B(ridge φ₀) BC={bc_B:.4f}'
          f'  C(mean b)   BC={bc_C:.4f}', flush=True)

    for strat, bc, rmse, p2, p3 in [
        ('A_datafit', bc_A, rmse_A, p2_A, p3_A),
        ('B_ridge',   bc_B, rmse_B, p2_B, p3_B),
        ('C_mean',    bc_C, rmse_C, p2_C, p3_C),
    ]:
        rows.append({'patient': pat, 'fold': fold, 'strategy': strat,
                     'bc': bc, 'rmse': rmse})

df = pd.DataFrame(rows)
df.to_csv(CR / 'loo_phi0_b_comparison.csv', index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n=== Mean LOO BC / RMSE by b_ho strategy ===')
for strat in ['A_datafit', 'B_ridge', 'C_mean']:
    sub = df[df.strategy == strat]
    label = {'A_datafit': 'A: data-fit (current, leaky)',
             'B_ridge':   'B: ridge from φ₀   (true OOS)',
             'C_mean':    'C: training mean    (naive)  '}[strat]
    print(f'  {label}: BC={sub.bc.mean():.4f}  RMSE={sub.rmse.mean():.4f}')

print('\nPer-patient BC (strategy A vs B):')
piv = df.pivot_table(index='patient', columns='strategy', values='bc')
piv['delta_B_vs_A'] = piv['B_ridge'] - piv['A_datafit']
piv['delta_C_vs_A'] = piv['C_mean']  - piv['A_datafit']
print(piv[['A_datafit', 'B_ridge', 'C_mean', 'delta_B_vs_A']].round(4).to_string())

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
colors = {'A_datafit': '#e07b54', 'B_ridge': '#4c78a8', 'C_mean': '#72b7b2'}
labels = {'A_datafit': 'A: data-fit (current)',
          'B_ridge':   'B: ridge from φ₀ (true OOS)',
          'C_mean':    'C: training mean'}

for ax, metric in zip(axes, ['bc', 'rmse']):
    x = np.arange(10)
    w = 0.26
    for j, strat in enumerate(['A_datafit', 'B_ridge', 'C_mean']):
        sub = df[df.strategy == strat].sort_values('fold')
        ax.bar(x + j*w, sub[metric], w, label=labels[strat],
               color=colors[strat], alpha=0.85)
        # mean line
        ax.axhline(sub[metric].mean(), color=colors[strat], linewidth=1.2,
                   linestyle='--', alpha=0.7)

    ax.set_xticks(x + w)
    ax.set_xticklabels(PATIENTS, fontsize=9)
    ax.set_xlabel('Patient', fontsize=11)
    ax.set_ylabel(metric.upper(), fontsize=11)
    ax.set_title(f'LOO {metric.upper()} by b_ho strategy', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
fig.savefig(CR / 'fig_loo_phi0_b_comparison.png', dpi=150)
fig.savefig(CR / 'fig_loo_phi0_b_comparison.pdf')
print('\nSaved fig_loo_phi0_b_comparison.png')
