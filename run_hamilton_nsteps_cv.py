#!/usr/bin/env python3
"""
run_hamilton_nsteps_cv.py — cross-validate nsteps for Hamilton+KEGG fit.

Tests nsteps in {25, 50, 100, 200, 500} on the full LOO-CV (8-fold) to find
the integration horizon that minimises held-out RMSE.

Each nsteps: fits A on 7 patients, predicts held-out patient.
Saves per-nsteps mean LOO-CV RMSE to nsteps_cv_hamilton_kegg.json.

Usage:
    python run_hamilton_nsteps_cv.py --gpu 1 [--nsteps 50 100 200]
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',    type=int, default=1)
parser.add_argument('--nsteps', type=int, nargs='+', default=[25, 50, 100, 200, 500])
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from guild_replicator_dieckow import GUILD_ORDER, N_G
from loo_cv_kegg_prior import build_net_flow, PHI_NPY

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp
from scipy.optimize import minimize as scipy_minimize

PATIENTS = list('ABCDEFGH')   # 8 patients (phi_guild.npy indices 0-7)
SIGMA    = 0.15
LAM      = 1e-4
N_EPOCHS = 2000
OUT      = _here / 'results' / 'dieckow_cr' / 'nsteps_cv_hamilton_kegg.json'

phi_all = np.load(PHI_NPY)
phi_sub = phi_all[:8]           # ABCDEFGH
print(f'phi_sub: {phi_sub.shape}  patients={PATIENTS}', flush=True)

net_flow = build_net_flow()
net_sym  = (net_flow + net_flow.T) / 2.0
n_p, n_sp = phi_sub.shape[0], N_G
n_A = n_sp * (n_sp + 1) // 2
diag_idx_np = np.array([j * (j + 1) // 2 + j for j in range(n_sp)])

# Warm start from existing KEGG fit
warm = _here / 'results' / 'dieckow_cr' / 'fit_glv_hamilton_kegg_prior.json'
d = json.load(open(warm))
A_warm = np.array(d['A'])
theta_A_warm = np.array([A_warm[i, j] for j in range(n_sp) for i in range(j + 1)])
b_warm = np.array(d['b_all'])

def _unpack_A(v):
    A = jnp.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            A = A.at[i, j].set(v[idx]); A = A.at[j, i].set(v[idx]); idx += 1
    return A

all_results = []

for nsteps in args.nsteps:
    print(f'\n{"="*50}', flush=True)
    print(f'nsteps={nsteps}', flush=True)
    t_ns = time.time()

    phi_obs_j = jnp.array(phi_sub)
    net_sym_j = jnp.array(net_sym)

    def _run_sim(theta, phi0):
        phibar = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=nsteps, dt=1e-4,
                                  phi_init=phi0, c_const=25.0, alpha_const=100.0)
        eq = phibar[-1]; s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

    def _pred2(theta_A, b_p, phi0):
        theta = jnp.concatenate([theta_A, b_p])
        phi2  = _run_sim(theta, phi0)
        phi3  = _run_sim(theta, phi2)
        return phi2, phi3

    def _loss_fn(theta_A, b_all, phi_obs_loc, net_sym_loc):
        n_p_loc = phi_obs_loc.shape[0]
        pred_all = jax.vmap(_pred2, in_axes=(None, 0, 0))
        phi2_all, phi3_all = pred_all(theta_A, b_all, phi_obs_loc[:, 0])
        sq   = jnp.sum((phi2_all - phi_obs_loc[:, 1])**2) + \
               jnp.sum((phi3_all - phi_obs_loc[:, 2])**2)
        rmse = jnp.sqrt(sq / (n_p_loc * 2 * n_sp))
        A    = _unpack_A(theta_A)
        sp_j = jnp.sign(net_sym_loc)
        mask = (sp_j != 0) & (~jnp.eye(n_sp, dtype=bool))
        pen  = jnp.where(mask,
                         jnp.abs(net_sym_loc) * jnp.maximum(0., -sp_j * A)**2 / (2*SIGMA**2),
                         0.).sum()
        return rmse + pen + LAM * jnp.sum(theta_A**2)

    # Compile once with full data
    print('  Compiling...', flush=True)
    _loss_grad = jax.jit(jax.value_and_grad(_loss_fn, argnums=(0, 1)))
    _ = _loss_grad(jnp.array(theta_A_warm), jnp.array(b_warm), phi_obs_j, net_sym_j)

    fold_rmses = []
    for hold in range(n_p):
        tr_idx  = [i for i in range(n_p) if i != hold]
        phi_tr  = jnp.array(phi_sub[tr_idx])
        phi_ho  = phi_sub[hold]
        net_tr  = net_sym_j

        n_tr = len(tr_idx)
        n_flat = n_A + n_tr * n_sp

        # warm start
        x0 = np.concatenate([theta_A_warm, b_warm[tr_idx].ravel()])

        def fg(x):
            ta = jnp.array(x[:n_A])
            ba = jnp.array(x[n_A:].reshape(n_tr, n_sp))
            loss, (gA, gb) = _loss_grad(ta, ba, phi_tr, net_tr)
            return float(loss), np.array(jnp.concatenate([gA, gb.ravel()]), dtype=np.float64)

        bounds = [(None, None)] * n_flat
        for k in diag_idx_np:
            bounds[k] = (None, 0.)

        res = scipy_minimize(fg, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': N_EPOCHS, 'ftol': 1e-12, 'gtol': 1e-7})

        # Fit b for held-out patient (b only, A fixed)
        ta_opt = jnp.array(res.x[:n_A])
        b_ho   = jnp.array(b_warm[hold])

        def fg_b(b_flat):
            b_j = jnp.array(b_flat)
            phi_ho_j = jnp.array(phi_ho[np.newaxis])
            loss, (_, gb) = _loss_grad(ta_opt, b_j[np.newaxis], phi_ho_j, net_tr)
            return float(loss), np.array(gb, dtype=np.float64)

        res_b = scipy_minimize(fg_b, np.array(b_ho), jac=True, method='L-BFGS-B',
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-7})
        b_ho_opt = jnp.array(res_b.x)

        # Evaluate on held-out
        theta = jnp.concatenate([ta_opt, b_ho_opt])
        phi2 = np.array(_run_sim(theta, jnp.array(phi_ho[0])))
        phi3 = np.array(_run_sim(theta, jnp.array(phi2)))
        rmse_ho = float(np.sqrt(np.mean(
            (phi2 - phi_ho[1])**2 / 2 + (phi3 - phi_ho[2])**2 / 2)))
        fold_rmses.append(rmse_ho)
        print(f'  fold {hold} (held={PATIENTS[hold]}): LOO-RMSE={rmse_ho:.4f}', flush=True)

    mean_rmse = float(np.mean(fold_rmses))
    print(f'  nsteps={nsteps}: LOO-CV mean={mean_rmse:.4f}  ({time.time()-t_ns:.0f}s)', flush=True)
    all_results.append({'nsteps': nsteps, 'loo_rmse_mean': mean_rmse,
                        'per_fold': fold_rmses})

json.dump(all_results, open(OUT, 'w'), indent=2)
print(f'\nSaved: {OUT}')
print('\n=== Summary ===')
for r in all_results:
    print(f"  nsteps={r['nsteps']:4d}: LOO-CV={r['loo_rmse_mean']:.4f}")
