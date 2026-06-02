#!/usr/bin/env python3
"""
run_hamilton_ss_loo.py — LOO-CV for Hamilton steady-state v2 model.

For each held-out patient:
  1. Train A on remaining 9 patients (damped replicator steady-state + 34-pair KEGG prior)
  2. Fit b for held-out patient (A fixed)
  3. Predict weeks 2,3 → LOO-CV RMSE

Usage:
    python run_hamilton_ss_loo.py --hold 0 --gpu 0   # fold 0 on GPU 0
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import argparse, json, sys, time
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--hold', type=int, required=True)
parser.add_argument('--gpu',  type=int, default=0)
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))

from guild_replicator_dieckow import GUILD_ORDER, N_G
from build_net_flow_expanded import build_net_flow_expanded
from hamilton_steadystate_jax import damped_iteration

PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
PATIENTS = list('ABCDEFGHKL')
SIGMA    = 0.15
LAM      = 1e-4
SS_ITER  = 2000
SS_DT    = 0.03
OUT_DIR  = _here / 'results' / 'dieckow_cr'

phi_all = np.load(PHI_NPY)   # (10,3,10)
n_p, n_sp = phi_all.shape[0], N_G
n_A = n_sp * (n_sp + 1) // 2

net_flow = build_net_flow_expanded(use_agora=True, verbose=False)
net_sym  = (net_flow + net_flow.T) / 2.0

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from scipy.optimize import minimize as scipy_minimize

hold = args.hold
tr_idx = [i for i in range(n_p) if i != hold]
phi_tr = phi_all[tr_idx]   # (9,3,10)
phi_ho = phi_all[hold]     # (3,10)
n_tr   = len(tr_idx)

print(f'LOO fold {hold}: held={PATIENTS[hold]}  train={[PATIENTS[i] for i in tr_idx]}', flush=True)

# Warm start from SS v2 fit
warm = OUT_DIR / 'fit_hamilton_kegg_steadystate_v2.json'
d    = json.load(open(warm))
A_w  = np.array(d['A'])
b_w  = np.array(d['b_all'])   # shape (10,10)

diag_idx_np = np.array([j * (j + 1) // 2 + j for j in range(n_sp)])

def _unpack_A(v):
    A = jnp.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            A = A.at[i, j].set(v[idx]); A = A.at[j, i].set(v[idx]); idx += 1
    return A

theta_A_warm = np.array([A_w[i, j] for j in range(n_sp) for i in range(j + 1)])
b_tr_warm    = b_w[tr_idx]    # (9,10)

phi_tr_j  = jnp.array(phi_tr)
net_sym_j = jnp.array(net_sym)

def _predict_ss(A, b_p, phi0):
    return damped_iteration(A, b_p, phi0, n_iter=SS_ITER, dt=SS_DT)

def _pred_two(theta_A, b_p, phi0):
    A    = _unpack_A(theta_A)
    phi2 = _predict_ss(A, b_p, phi0)
    phi3 = _predict_ss(A, b_p, phi2)
    return phi2, phi3

_pred_all = jax.vmap(_pred_two, in_axes=(None, 0, 0))

@jax.jit
def loss_fn(theta_A, b_tr):
    phi2_all, phi3_all = _pred_all(theta_A, b_tr, phi_tr_j[:, 0])
    sq   = jnp.sum((phi2_all - phi_tr_j[:, 1])**2) + \
           jnp.sum((phi3_all - phi_tr_j[:, 2])**2)
    rmse = jnp.sqrt(sq / (n_tr * 2 * n_sp))
    A    = _unpack_A(theta_A)
    sp_j = jnp.sign(net_sym_j)
    mask = (sp_j != 0) & (~jnp.eye(n_sp, dtype=bool))
    pen  = jnp.where(mask,
                     jnp.abs(net_sym_j) * jnp.maximum(0., -sp_j * A)**2 / (2*SIGMA**2),
                     0.).sum()
    return rmse + pen + LAM * jnp.sum(theta_A**2)

n_flat = n_A + n_tr * n_sp

@jax.jit
def _lag(x_flat):
    ta = x_flat[:n_A]; ba = x_flat[n_A:].reshape(n_tr, n_sp)
    loss, (gA, gb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(ta, ba)
    return loss, jnp.concatenate([gA, gb.ravel()])

print('Compiling...', flush=True)
x0 = np.concatenate([theta_A_warm, b_tr_warm.ravel()])
init_loss = float(loss_fn(jnp.array(theta_A_warm), jnp.array(b_tr_warm)))
print(f'Initial loss={init_loss:.5f}', flush=True)

call_count = [0]; t0 = time.time()

def fg(x):
    call_count[0] += 1
    loss, grad = _lag(jnp.array(x))
    if call_count[0] % 100 == 1:
        print(f'  iter {call_count[0]:4d}  loss={float(loss):.5f}  ({time.time()-t0:.1f}s)',
              flush=True)
    return float(loss), np.array(grad, dtype=np.float64)

bounds = [(None, None)] * n_flat
for k in diag_idx_np:
    bounds[k] = (None, 0.)

res = scipy_minimize(fg, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-7})
print(f'  {res.message}  iters={call_count[0]}  loss={res.fun:.5f}', flush=True)

ta_opt = jnp.array(res.x[:n_A])
A_opt  = _unpack_A(ta_opt)

# Fit b for held-out patient
b_ho_warm = jnp.array(b_w[hold])

@jax.jit
def loss_b(b_p):
    phi_ho_j = jnp.array(phi_ho)
    phi2 = _predict_ss(A_opt, b_p, phi_ho_j[0])
    phi3 = _predict_ss(A_opt, b_p, phi2)
    sq = jnp.sum((phi2 - phi_ho_j[1])**2) + jnp.sum((phi3 - phi_ho_j[2])**2)
    return jnp.sqrt(sq / (2 * n_sp))

def fg_b(b):
    bj = jnp.array(b)
    loss, grad = jax.value_and_grad(loss_b)(bj)
    return float(loss), np.array(grad, dtype=np.float64)

res_b = scipy_minimize(fg_b, np.array(b_ho_warm), jac=True, method='L-BFGS-B',
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-7})
b_ho_opt = jnp.array(res_b.x)

# Evaluate
phi_ho_j = jnp.array(phi_ho)
phi2 = np.array(_predict_ss(A_opt, b_ho_opt, phi_ho_j[0]))
phi3 = np.array(_predict_ss(A_opt, b_ho_opt, jnp.array(phi2)))
rmse = float(np.sqrt((np.mean((phi2 - phi_ho[1])**2) + np.mean((phi3 - phi_ho[2])**2)) / 2))

sp_mat  = np.sign(net_sym)
mask_np = (sp_mat != 0) & (~np.eye(n_sp, dtype=bool))
A_np    = np.array(A_opt)
n_agree = int(((np.sign(A_np) == sp_mat) & mask_np).sum())
n_tot   = int(mask_np.sum())

print(f'\nFold {hold} ({PATIENTS[hold]}): LOO-RMSE={rmse:.4f}  SA={n_agree}/{n_tot}', flush=True)

out = {'patient': PATIENTS[hold], 'hold_idx': hold, 'rmse': rmse,
       'sign_agreement': f'{n_agree}/{n_tot}',
       'model': f'Hamilton SS v2 LOO (34 pairs, 10 patients, SS_ITER={SS_ITER})'}
json.dump(out, open(OUT_DIR / f'loo_ss_v2_fold{hold}.json', 'w'), indent=2)
print(f'Saved loo_ss_v2_fold{hold}.json', flush=True)
