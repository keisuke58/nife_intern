#!/usr/bin/env python3
"""
eval_floor_bc.py — evaluate BC improvement from minimum-abundance floor.

Post-processing approach: after ODE prediction, clip phi_i >= eps and renormalize.
No re-training needed. Tests eps in [0, 0.005, 0.01, 0.02, 0.03, 0.05].

Usage:
    /home/nishioka/IKM_Hiwi/.venv_jax/bin/python eval_floor_bc.py
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from hamilton_ode_jax_nsp import simulate_0d_nsp
from guild_replicator_dieckow import GUILD_ORDER, N_G

CR       = HERE / 'results' / 'dieckow_cr'
OTU      = HERE / 'results' / 'dieckow_otu'
PATIENTS = list('ABCDEFGHKL')
EPS_VALS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05]

phi_all = np.load(OTU / 'phi_guild.npy')   # (10, 3, 10) patients × weeks × guilds


def bray_curtis(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return 1.0 - 2.0 * np.minimum(x, y).sum() / (x.sum() + y.sum())


def apply_floor(phi, eps):
    """Clip each guild to >= eps, then renormalize."""
    if eps == 0.0:
        return phi
    phi_f = np.maximum(phi, eps)
    return phi_f / phi_f.sum()


def pack_theta_A(A_mat):
    n = len(A_mat)
    return jnp.array([A_mat[i][j] for j in range(n) for i in range(j + 1)])


def run_loo_prediction(fold_json):
    """Load fold JSON, run ODE, return (pred_wk2, pred_wk3, patient_idx)."""
    d = json.load(open(fold_json))
    if 'A' not in d:
        return None
    p_idx    = PATIENTS.index(d['patient'])
    theta_A  = pack_theta_A(np.array(d['A']))
    b_ho     = jnp.array(d['b_ho'])
    theta    = jnp.concatenate([theta_A, b_ho])
    phi0     = jnp.array(phi_all[p_idx, 0])

    @jax.jit
    def step(phi):
        eq = simulate_0d_nsp(theta, n_sp=N_G, n_steps=100, dt=1e-4,
                             phi_init=phi, c_const=25.0, alpha_const=100.0)[-1]
        s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(N_G) / N_G)

    p2 = np.array(step(phi0))
    p3 = np.array(step(jnp.array(p2)))
    return p2, p3, p_idx


# ── Collect predictions for each model × fold ────────────────────────────────
MODELS = {
    'Hamilton (no prior)':   'loo_expanded_a0p0_fold{fold}.json',
    'Hamilton α=0.25':       'loo_expanded_a0p25_fold{fold}.json',
    'Hamilton α=0.5':        'loo_expanded_a0p5_fold{fold}.json',
    'AGORA α=0.0':           'loo_expanded_agora_a0p0_fold{fold}.json',
    'AGORA α=0.25':          'loo_expanded_agora_a0p25_fold{fold}.json',
}

rows = []
for model_name, pattern in MODELS.items():
    print(f'\n{model_name}', flush=True)
    preds = []   # (fold, wk) → (pred, obs)
    for fold in range(10):
        path = CR / pattern.format(fold=fold)
        if not path.exists():
            continue
        result = run_loo_prediction(path)
        if result is None:
            continue
        p2, p3, p_idx = result
        obs2, obs3 = phi_all[p_idx, 1], phi_all[p_idx, 2]
        preds.append((fold, p2, p3, obs2, obs3))
        print(f'  fold {fold} ok', end='', flush=True)
    print()

    for eps in EPS_VALS:
        bc_vals = []
        for fold, p2, p3, obs2, obs3 in preds:
            p2f = apply_floor(p2, eps)
            p3f = apply_floor(p3, eps)
            bc_vals.append(bray_curtis(p2f, obs2))
            bc_vals.append(bray_curtis(p3f, obs3))
        mean_bc = float(np.mean(bc_vals))
        rows.append({'model': model_name, 'eps': eps, 'bc_mean': mean_bc,
                     'n_obs': len(bc_vals)})
        print(f'  eps={eps:.3f}  BC={mean_bc:.4f}')

df = pd.DataFrame(rows)
df.to_csv(CR / 'floor_bc_sweep.csv', index=False)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))
colors = plt.cm.tab10.colors
for i, model_name in enumerate(MODELS):
    sub = df[df.model == model_name].sort_values('eps')
    if sub.empty:
        continue
    ax.plot(sub.eps * 100, sub.bc_mean, 'o-', color=colors[i],
            label=model_name, linewidth=1.8, markersize=5)
    # annotate improvement from eps=0
    bc0  = sub[sub.eps == 0.0].bc_mean.values
    bc_best = sub.bc_mean.min()
    if len(bc0):
        delta = bc0[0] - bc_best
        ax.annotate(f'−{delta:.3f}', xy=(sub.loc[sub.bc_mean.idxmin(), 'eps'] * 100, bc_best),
                    xytext=(4, 4), textcoords='offset points', fontsize=7,
                    color=colors[i])

ax.set_xlabel('Floor ε (%)', fontsize=11)
ax.set_ylabel('Mean LOO Bray-Curtis', fontsize=11)
ax.set_title('BC improvement from minimum-abundance floor', fontsize=12)
ax.legend(fontsize=8, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(CR / 'fig_floor_bc_sweep.png', dpi=150)
fig.savefig(CR / 'fig_floor_bc_sweep.pdf')
print(f'\nSaved fig_floor_bc_sweep.png')

# ── Best eps per model ────────────────────────────────────────────────────────
print('\n=== Best eps per model ===')
for model_name in MODELS:
    sub = df[df.model == model_name]
    if sub.empty:
        continue
    best = sub.loc[sub.bc_mean.idxmin()]
    bc0  = sub[sub.eps == 0.0].bc_mean.values[0]
    print(f'  {model_name:22s}: BC(ε=0)={bc0:.4f}  best BC={best.bc_mean:.4f}'
          f'  at ε={best.eps:.3f}  Δ=−{bc0-best.bc_mean:.4f}')
