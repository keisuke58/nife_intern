#!/usr/bin/env python3
"""
run_random_baseline.py — Random sign permutation baseline (JAX only, no cobra).
Launched as subprocess from run_validation_analyses.py.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',      type=int,   default=2)
parser.add_argument('--n-random', type=int,   default=100)
parser.add_argument('--out',      type=str,   required=True)
args = parser.parse_args()

import os
os.environ['CUDA_VISIBLE_DEVICES']           = str(args.gpu)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']  = 'false'   # don't hog all GPU memory

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)
import optax

from guild_replicator_dieckow import GUILD_ORDER, N_G
from build_net_flow_expanded import build_net_flow_expanded

PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
PATIENTS = list('ABCDEFGHKL')
N_P      = 10
SIGMA    = 0.15
LAM      = 1e-4

phi_obs_np = np.load(PHI_NPY)
phi_obs    = jnp.array(phi_obs_np)

net_agora     = build_net_flow_expanded(use_agora=True, verbose=False, agora_weight=1.0)
net_sym_agora = (net_agora + net_agora.T) / 2.0
sign_agora    = np.sign(net_sym_agora)

n_sp = N_G
n_A  = n_sp * (n_sp + 1) // 2
triu_r, triu_c = np.tril_indices(n_sp)

pairs_constrained = [(i, j) for i in range(n_sp) for j in range(i+1, n_sp)
                     if sign_agora[i, j] != 0]
print(f'  Constrained pairs: {len(pairs_constrained)}', flush=True)

def make_A(theta_A):
    A = jnp.zeros((n_sp, n_sp))
    A = A.at[triu_r, triu_c].set(theta_A)
    return A + A.T - jnp.diag(jnp.diag(A))

@jax.jit
def glv_step(phi, Ab):
    """One week of gLV replicator dynamics (Euler, 50 microsteps)."""
    A, b = Ab
    def body(phi, _):
        f = b + A @ phi
        f_bar = jnp.dot(phi, f)
        phi = phi + 0.02 * phi * (f - f_bar)
        phi = jnp.clip(phi, 0.0, None)
        s = phi.sum()
        return jnp.where(s > 1e-12, phi / s, phi), None
    phi_out, _ = jax.lax.scan(body, phi, None, length=50)
    return phi_out

def pred_week(theta_A, b_p, phi0):
    A  = make_A(theta_A)
    Ab = (A, b_p)
    phi2 = glv_step(phi0, Ab)
    phi3 = glv_step(phi2, Ab)
    return phi2, phi3

_pred_all = jax.jit(jax.vmap(pred_week, in_axes=(None, 0, 0)))

def fit_glv(sign_mat_np, n_steps=300, seed=0):
    sign_mat_j = jnp.array(sign_mat_np)
    mask = (sign_mat_np != 0) & (~np.eye(n_sp, dtype=bool))
    rng  = np.random.default_rng(seed)
    theta_A = jnp.array(rng.normal(0, 0.1, n_A))
    b_all   = jnp.array(np.full((N_P, n_sp), 0.1))

    @jax.jit
    def loss(theta_A, b_all):
        A = make_A(theta_A)
        phi2_all, phi3_all = _pred_all(theta_A, b_all, phi_obs[:, 0])
        sq = (jnp.sum((phi2_all - phi_obs[:, 1])**2) +
              jnp.sum((phi3_all - phi_obs[:, 2])**2))
        sgn_viol = jnp.where(
            mask,
            jnp.maximum(0.0, -sign_mat_j * A)**2 / (2 * SIGMA**2),
            0.0)
        return sq + jnp.sum(sgn_viol) + LAM * jnp.sum(theta_A**2)

    opt   = optax.adam(1e-3)
    state = opt.init((theta_A, b_all))
    for _ in range(n_steps):
        grads = jax.grad(loss, argnums=(0, 1))(theta_A, b_all)
        updates, state = opt.update(grads, state)
        theta_A = theta_A + updates[0]
        b_all   = b_all   + updates[1]

    A_fit  = np.array(make_A(theta_A))
    phi2, phi3 = _pred_all(theta_A, b_all, phi_obs[:, 0])
    rmse = float(jnp.sqrt(jnp.mean((phi2 - phi_obs[:, 1])**2 +
                                    (phi3 - phi_obs[:, 2])**2)))
    pairs = [(i, j) for i in range(n_sp) for j in range(i+1, n_sp)
             if sign_mat_np[i, j] != 0]
    sa = float(np.mean([np.sign(A_fit[i,j]) == sign_mat_np[i,j]
                        for i, j in pairs])) if pairs else float('nan')
    return rmse, sa

# AGORA reference
print('  Fitting AGORA W=1.0 reference...', flush=True)
rmse_agora, sa_agora = fit_glv(sign_agora, n_steps=500)
print(f'  AGORA: RMSE={rmse_agora:.4f}  SA={sa_agora:.1%}', flush=True)

# Random permutations
print(f'\n  Running {args.n_random} random permutations...', flush=True)
rng = np.random.default_rng(42)
random_rmses, random_sas = [], []
t0 = time.time()

for k in range(args.n_random):
    rand_sign = np.zeros((n_sp, n_sp))
    for i, j in pairs_constrained:
        s = rng.choice([-1, 1])
        rand_sign[i, j] = s
        rand_sign[j, i] = s
    rmse_r, sa_r = fit_glv(rand_sign, n_steps=300, seed=k)
    random_rmses.append(rmse_r)
    random_sas.append(sa_r)
    if (k+1) % 10 == 0:
        print(f'  {k+1:3d}/{args.n_random}  mean RMSE={np.mean(random_rmses):.4f}  '
              f't={time.time()-t0:.0f}s', flush=True)

random_rmses = np.array(random_rmses)
random_sas   = np.array(random_sas)
pct_better   = float((random_rmses > rmse_agora).mean()) * 100

print(f'\n  === Random baseline results ===', flush=True)
print(f'  AGORA RMSE:  {rmse_agora:.4f}  SA={sa_agora:.1%}', flush=True)
print(f'  Random RMSE: {random_rmses.mean():.4f} ± {random_rmses.std():.4f}', flush=True)
print(f'  Random SA:   {random_sas.mean():.1%} ± {random_sas.std():.1%}  (expected ~50%)', flush=True)
print(f'  AGORA beats {pct_better:.0f}% of random permutations', flush=True)

# Merge with existing FBA results and save
out_path = Path(args.out)
existing = json.load(open(out_path)) if out_path.exists() else {}
existing['random_baseline'] = {
    'n_random':         args.n_random,
    'agora_rmse':       float(rmse_agora),
    'agora_sa':         float(sa_agora),
    'random_rmse_mean': float(random_rmses.mean()),
    'random_rmse_std':  float(random_rmses.std()),
    'random_sa_mean':   float(random_sas.mean()),
    'random_sa_std':    float(random_sas.std()),
    'pct_agora_better': float(pct_better),
    'random_rmses':     random_rmses.tolist(),
    'random_sas':       random_sas.tolist(),
}
with open(out_path, 'w') as f:
    json.dump(existing, f, indent=2)
print(f'\nSaved → {out_path}', flush=True)
