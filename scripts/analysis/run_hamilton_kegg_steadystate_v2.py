#!/usr/bin/env python3
"""
run_hamilton_kegg_steadystate.py — Hamilton replicator fit via steady-state prediction.

Instead of integrating the ODE for nsteps, predicts the equilibrium φ* of
    φ̇_i = φ_i(b_i + Σ_j A_ij φ_j − f̄) = 0
using damped replicator iteration (fully JAX-differentiable).

This eliminates the nsteps dependency: A and b are optimized to place the
steady state at the observed week-2/3 values (starting from week 1).

Usage:
    python run_hamilton_kegg_steadystate.py [--gpu GPU] [--no-agora]
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import argparse, json, sys, time
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',      type=int, default=3)
parser.add_argument('--no-agora', action='store_true')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))

from guild_replicator_dieckow import GUILD_ORDER, N_G
from build_net_flow_expanded import build_net_flow_expanded

PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
PATIENTS = list('ABCDEFGHKL')   # full cohort
SIGMA    = 0.15
LAM      = 1e-4
SS_ITER  = 2000
SS_DT    = 0.03
OUT      = _here / 'results' / 'dieckow_cr' / 'fit_hamilton_kegg_steadystate_v2.json'

# ── Load data ─────────────────────────────────────────────────────────────────
phi_all = np.load(PHI_NPY)         # (10, 3, 10)
phi_sub = phi_all                   # all 10 patients
print(f'Loaded phi_sub: {phi_sub.shape}  patients={PATIENTS}', flush=True)
n_p = phi_sub.shape[0]

# ── Expanded flow matrix ──────────────────────────────────────────────────────
use_agora = not args.no_agora
print(f'Building expanded flow matrix (use_agora={use_agora})...', flush=True)
net_flow = build_net_flow_expanded(use_agora=use_agora, verbose=True)
net_sym  = (net_flow + net_flow.T) / 2.0
sp_mat   = np.sign(net_sym)
mask_np  = (sp_mat != 0) & (~np.eye(N_G, dtype=bool))
n_pairs  = int(mask_np.sum() // 2)
print(f'Expanded flow: {n_pairs} undirected constrained pairs', flush=True)

# ── JAX setup ─────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_steadystate_jax import damped_iteration

n_sp = N_G
n_A  = n_sp * (n_sp + 1) // 2

# ── Warm start ────────────────────────────────────────────────────────────────
warm = _here / 'results' / 'dieckow_cr' / 'fit_glv_hamilton_kegg_prior.json'
if warm.exists():
    d        = json.load(open(warm))
    A_full   = np.array(d['A'])[:n_sp, :n_sp]
    theta_A0 = np.array([A_full[i, j] for j in range(n_sp) for i in range(j + 1)])
    b_raw    = np.array(d['b_all'])
    if b_raw.ndim == 2 and b_raw.shape[0] >= n_p and b_raw.shape[1] >= n_sp:
        b_all0 = b_raw[:n_p, :n_sp].copy()
    else:
        b_all0 = np.full((n_p, n_sp), 0.1)
    print(f'Warm-start from {warm.name}', flush=True)
else:
    diag_idx0        = np.array([j * (j + 1) // 2 + j for j in range(n_sp)])
    theta_A0         = np.zeros(n_A)
    theta_A0[diag_idx0] = -0.1
    b_all0           = np.full((n_p, n_sp), 0.1)

theta_A   = jnp.array(theta_A0)
b_all     = jnp.array(b_all0)
phi_obs   = jnp.array(phi_sub)
net_sym_j = jnp.array(net_sym)
diag_idx_np = np.array([j * (j + 1) // 2 + j for j in range(n_sp)])

# ── Steady-state prediction ───────────────────────────────────────────────────
def _unpack_A(v):
    A = jnp.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            A = A.at[i, j].set(v[idx]); A = A.at[j, i].set(v[idx]); idx += 1
    return A

def _predict_ss(A, b_p, phi0):
    """Predict steady state from phi0 using damped iteration."""
    return damped_iteration(A, b_p, phi0, n_iter=SS_ITER, dt=SS_DT)

def _pred_two_weeks_ss(theta_A, b_p, phi0):
    A    = _unpack_A(theta_A)
    phi2 = _predict_ss(A, b_p, phi0)
    phi3 = _predict_ss(A, b_p, phi2)
    return phi2, phi3

_pred_all_ss = jax.vmap(_pred_two_weeks_ss, in_axes=(None, 0, 0))

@jax.jit
def loss_fn(theta_A, b_all):
    phi2_all, phi3_all = _pred_all_ss(theta_A, b_all, phi_obs[:, 0])
    sq   = jnp.sum((phi2_all - phi_obs[:, 1]) ** 2) + jnp.sum((phi3_all - phi_obs[:, 2]) ** 2)
    rmse = jnp.sqrt(sq / (n_p * 2 * n_sp))
    A    = _unpack_A(theta_A)
    sp_j = jnp.sign(net_sym_j)
    mask = (sp_j != 0) & (~jnp.eye(n_sp, dtype=bool))
    pen  = jnp.where(mask,
                     jnp.abs(net_sym_j) * jnp.maximum(0.0, -sp_j * A) ** 2 / (2 * SIGMA ** 2),
                     0.0).sum()
    return rmse + pen + LAM * jnp.sum(theta_A ** 2)

# ── Compile ───────────────────────────────────────────────────────────────────
from scipy.optimize import minimize as scipy_minimize

n_flat = n_A + n_p * n_sp

@jax.jit
def _loss_and_grad_flat(x_flat):
    ta = x_flat[:n_A]
    ba = x_flat[n_A:].reshape(n_p, n_sp)
    loss, (gA, gb) = jax.value_and_grad(loss_fn, argnums=(0, 1))(ta, ba)
    return loss, jnp.concatenate([gA, gb.ravel()])

print(f'Compiling Hamilton steady-state (n_sp={n_sp}, SS_ITER={SS_ITER})...', flush=True)
x0 = np.concatenate([np.array(theta_A), np.array(b_all).ravel()])
init_loss = float(loss_fn(theta_A, b_all))
print(f'Initial loss={init_loss:.5f}', flush=True)

call_count = [0]
t0 = time.time()

def fg(x):
    call_count[0] += 1
    xj   = jnp.array(x)
    loss, grad = _loss_and_grad_flat(xj)
    if call_count[0] % 50 == 1:
        print(f'  iter {call_count[0]:4d}  loss={float(loss):.5f}  ({time.time()-t0:.1f}s)',
              flush=True)
    return float(loss), np.array(grad, dtype=np.float64)

bounds = [(None, None)] * n_flat
for k in diag_idx_np:
    bounds[k] = (None, 0.0)

res = scipy_minimize(fg, x0, jac=True, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 5000, 'ftol': 1e-12, 'gtol': 1e-7})
print(f'L-BFGS-B: {res.message}  iters={call_count[0]}  loss={res.fun:.5f}', flush=True)

# ── Evaluate ──────────────────────────────────────────────────────────────────
x_opt   = res.x
theta_A = jnp.array(x_opt[:n_A])
b_all   = jnp.array(x_opt[n_A:].reshape(n_p, n_sp))
A_opt   = np.array(_unpack_A(theta_A))
b_opt   = np.array(b_all)

sq, cnt, all_obs, all_pred = 0.0, 0, [], []
for p in range(n_p):
    A_j  = jnp.array(A_opt)
    phi2 = np.array(_predict_ss(A_j, b_all[p], phi_obs[p, 0]))
    phi3 = np.array(_predict_ss(A_j, b_all[p], jnp.array(phi2)))
    sq  += float(np.sum((phi2 - phi_sub[p, 1]) ** 2) + np.sum((phi3 - phi_sub[p, 2]) ** 2))
    cnt += 2 * n_sp
    all_obs  += phi_sub[p, 1].tolist() + phi_sub[p, 2].tolist()
    all_pred += phi2.tolist() + phi3.tolist()

rmse = float(np.sqrt(sq / cnt))
r    = float(np.corrcoef(all_obs, all_pred)[0, 1])

n_agree = int(((np.sign(A_opt) == sp_mat) & mask_np).sum())
n_tot   = int(mask_np.sum())
print(f'Hamilton SS: RMSE={rmse:.4f}, r={r:.4f}  ({time.time()-t0:.1f}s)', flush=True)
print(f'Sign agreement: {n_agree}/{n_tot} ({100*n_agree/n_tot:.0f}%)', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'rmse': rmse, 'r': r,
    'sign_agreement': f'{n_agree}/{n_tot}',
    'n_constrained_pairs': n_pairs,
    'patients': PATIENTS, 'guilds': GUILD_ORDER,
    'A': A_opt.tolist(), 'b_all': b_opt.tolist(),
    'net_flow': net_flow.tolist(),
    'model': (f'Hamilton SS damped-iter ({n_p} pat, {N_G} guild, '
              f'sigma={SIGMA}, ss_iter={SS_ITER}, dt={SS_DT}, agora={use_agora})'),
}
json.dump(out, open(OUT, 'w'), indent=2)
print(f'Saved: {OUT}', flush=True)
