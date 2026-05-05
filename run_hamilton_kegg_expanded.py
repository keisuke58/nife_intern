#!/usr/bin/env python3
"""
run_hamilton_kegg_expanded.py — Hamilton ODE fit with expanded KEGG/Szafranski flow matrix.

Uses build_net_flow_expanded (L1+L2 Szafranski + L3 AGORA FBA) instead of the
original 11-pair flow matrix.  Saves to fit_glv_hamilton_kegg_expanded.json.

Usage:
    python run_hamilton_kegg_expanded.py [--gpu GPU_ID] [--nsteps N] [--no-agora]
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',        type=int,   default=2)
parser.add_argument('--nsteps',     type=int,   default=100)
parser.add_argument('--no-agora',   action='store_true')
parser.add_argument('--w-agora',    type=float, default=1.0,  help='AGORA L3 sign-prior weight')
parser.add_argument('--prior-mode', type=str,   default='sign',
                    choices=['sign', 'macarthur'],
                    help='sign: current sign penalty; macarthur: Gaussian prior from FBA Phi')
parser.add_argument('--c-agora',    type=float, default=0.1,
                    help='[macarthur mode] scaling constant c in A~N(c·Phi, sigma²)')
parser.add_argument('--sigma-mac',  type=float, default=0.5,
                    help='[macarthur mode] prior std sigma')
parser.add_argument('--out-suffix', type=str,   default='',   help='suffix for output filename')
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from guild_replicator_dieckow import GUILD_ORDER, N_G
from build_net_flow_expanded import build_net_flow_expanded

PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
PATIENTS = list('ABCDEFGHKL')   # full cohort (all 10 patients)
SIGMA    = 0.15
if args.out_suffix:
    _suffix = args.out_suffix
elif args.no_agora:
    _suffix = '_no_agora'
elif args.prior_mode == 'macarthur':
    _suffix = f'_mac_c{args.c_agora:.3f}'.replace('.', 'p')
else:
    _suffix = f'_agora_w{args.w_agora:.1f}'.replace('.', 'p')
OUT      = _here / 'results' / 'dieckow_cr' / f'fit_glv_hamilton_kegg_expanded{_suffix}.json'

# ── Load data ─────────────────────────────────────────────────────────────────
phi_all = np.load(PHI_NPY)
phi_sub = phi_all                                # all 10 patients
print(f'Loaded phi_sub: {phi_sub.shape}  patients={PATIENTS}', flush=True)

# ── Expanded flow matrix ──────────────────────────────────────────────────────
use_agora = not args.no_agora
print(f'Building expanded flow matrix (use_agora={use_agora}, w_agora={args.w_agora})...', flush=True)
net_flow = build_net_flow_expanded(use_agora=use_agora, verbose=True,
                                    agora_weight=args.w_agora)

net_sym  = (net_flow + net_flow.T) / 2.0
sp_mat   = np.sign(net_sym)
mask_np  = (sp_mat != 0) & (~np.eye(N_G, dtype=bool))
n_pairs  = int(mask_np.sum() // 2)
print(f'Expanded flow: {n_pairs} undirected constrained pairs', flush=True)

# ── JAX setup ─────────────────────────────────────────────────────────────────
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
from hamilton_ode_jax_nsp import simulate_0d_nsp

n_p  = phi_sub.shape[0]
n_sp = N_G
n_A  = n_sp * (n_sp + 1) // 2
LAM  = 1e-4

# ── Warm start ────────────────────────────────────────────────────────────────
warm = _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json'
if warm.exists():
    d      = json.load(open(warm))
    A_full = np.array(d['A'])[:n_sp, :n_sp]
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

theta_A  = jnp.array(theta_A0)
b_all    = jnp.array(b_all0)
phi_obs  = jnp.array(phi_sub)
net_sym_j = jnp.array(net_sym)
diag_idx_np = np.array([j * (j + 1) // 2 + j for j in range(n_sp)])

nsteps = args.nsteps

# ── MacArthur prior (optional) ────────────────────────────────────────────────
PRIOR_MODE = args.prior_mode
C_AGORA    = args.c_agora
SIGMA_MAC  = args.sigma_mac
phi_mac_j  = None
mac_mask_j = None

if PRIOR_MODE == 'macarthur' and use_agora:
    from guild_agora_signs import get_agora_phi_matrix
    agora_dir = _here / 'data' / 'homd_db' / 'agora_gems'
    print(f'Computing MacArthur Phi matrix (c={C_AGORA}, sigma={SIGMA_MAC})...', flush=True)
    phi_mac_np, mac_mask_np = get_agora_phi_matrix(agora_dir, verbose=True)
    phi_mac_j  = jnp.array(phi_mac_np)
    mac_mask_j = jnp.array(mac_mask_np)
    print(f'Phi range: [{phi_mac_np.min():.3f}, {phi_mac_np.max():.3f}]  '
          f'non-zero pairs: {int(mac_mask_np.sum())}', flush=True)

# ── ODE helpers ───────────────────────────────────────────────────────────────
def _run_sim(theta, phi_init):
    phibar = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=nsteps, dt=1e-4,
                              phi_init=phi_init, c_const=25.0, alpha_const=100.0)
    eq = phibar[-1]; s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

def _pred_two_weeks(theta_A, b_p, phi0):
    theta = jnp.concatenate([theta_A, b_p])
    phi2  = _run_sim(theta, phi0)
    phi3  = _run_sim(theta, phi2)
    return phi2, phi3

_pred_all = jax.vmap(_pred_two_weeks, in_axes=(None, 0, 0))

@jax.jit
def pred_week(theta_A, b, phi0):
    return _run_sim(jnp.concatenate([theta_A, b]), phi0)

def _unpack_A(v):
    A = jnp.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j + 1):
            A = A.at[i, j].set(v[idx]); A = A.at[j, i].set(v[idx]); idx += 1
    return A

@jax.jit
def loss_fn(theta_A, b_all):
    phi2_all, phi3_all = _pred_all(theta_A, b_all, phi_obs[:, 0])
    sq   = jnp.sum((phi2_all - phi_obs[:, 1]) ** 2) + jnp.sum((phi3_all - phi_obs[:, 2]) ** 2)
    rmse = jnp.sqrt(sq / (n_p * 2 * n_sp))
    A    = _unpack_A(theta_A)

    if PRIOR_MODE == 'macarthur' and phi_mac_j is not None:
        # MacArthur Gaussian prior: A[i,j] ~ N(c · Phi[i,j], sigma²)
        # L = Σ_{i≠j,mask} (A[i,j] − c·Phi[i,j])² / (2σ²)
        pen = jnp.where(
            mac_mask_j,
            (A - C_AGORA * phi_mac_j) ** 2 / (2.0 * SIGMA_MAC ** 2),
            0.0
        ).sum()
    else:
        # Sign prior (default): penalise wrong sign weighted by flow magnitude
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

print(f'Compiling Hamilton JAX (n_sp={n_sp}, nsteps={nsteps})...', flush=True)
x0 = np.concatenate([np.array(theta_A), np.array(b_all).ravel()])
init_loss = float(loss_fn(theta_A, b_all))
print(f'Initial loss={init_loss:.5f}', flush=True)

call_count = [0]
t0 = time.time()

def fg(x):
    call_count[0] += 1
    xj = jnp.array(x)
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
    phi2 = np.array(pred_week(theta_A, b_all[p], phi_obs[p, 0]))
    phi3 = np.array(pred_week(theta_A, b_all[p], jnp.array(phi2)))
    sq  += float(np.sum((phi2 - phi_sub[p, 1]) ** 2) + np.sum((phi3 - phi_sub[p, 2]) ** 2))
    cnt += 2 * n_sp
    all_obs  += phi_sub[p, 1].tolist() + phi_sub[p, 2].tolist()
    all_pred += phi2.tolist() + phi3.tolist()

rmse = float(np.sqrt(sq / cnt))
r    = float(np.corrcoef(all_obs, all_pred)[0, 1])

n_agree = int(((np.sign(A_opt) == sp_mat) & mask_np).sum())
n_tot   = int(mask_np.sum())
print(f'Hamilton expanded: RMSE={rmse:.4f}, r={r:.4f}  ({time.time()-t0:.1f}s)', flush=True)
print(f'Sign agreement: {n_agree}/{n_tot} ({100*n_agree/n_tot:.0f}%)', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
out = {
    'rmse': rmse, 'r': r,
    'sign_agreement': f'{n_agree}/{n_tot}',
    'n_constrained_pairs': n_pairs,
    'patients': PATIENTS, 'guilds': GUILD_ORDER,
    'A': A_opt.tolist(), 'b_all': b_opt.tolist(),
    'net_flow': net_flow.tolist(),
    'model': (f'Hamilton JAX L-BFGS-B expanded-prior ({n_p} pat, {N_G} guild, '
              f'sigma={SIGMA}, nsteps={nsteps}, agora={use_agora})'),
}
json.dump(out, open(OUT, 'w'), indent=2)
print(f'Saved: {OUT}', flush=True)
