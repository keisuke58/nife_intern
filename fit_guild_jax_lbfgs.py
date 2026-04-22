#!/usr/bin/env python3
"""
10-guild gLV fit: JAX autodiff + scipy L-BFGS-B.

Improvement over fit_guild_replicator.py:
  - jax.lax.scan-based RK4 → fast JIT compilation, no loop unrolling
  - Exact gradient via jax.value_and_grad (~14ms/call vs 2800ms FD)
  - Checkpoint saved after every start (safe against walltime kill)
  - 4 starts (warm + 3 rand), maxiter=2000 → estimated < 30 min total

Output: results/dieckow_cr/fit_guild.json (overwrites if RMSE improves)
"""

import json, sys, time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent))
from guild_replicator_dieckow import GUILD_ORDER, N_G, pack, unpack, default_A

RESULTS_DIR = Path(__file__).parent / 'results' / 'dieckow_cr'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PHI_NPY    = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'
LAMBDA_REG = 1e-4
N_STEPS    = 100   # fixed RK4 steps per week; scan-based (O(1) compile time)
PATIENTS   = list('ABCDEFGHKL')
N_A        = N_G * N_G


# ---- JAX replicator ODE via lax.scan (O(1) compile regardless of N_STEPS) ----

@jax.jit
def _rk4_step(phi, b, A, h):
    def rhs(p):
        f = b + A @ p
        return p * (f - p @ f)
    k1 = rhs(phi)
    k2 = rhs(phi + 0.5 * h * k1)
    k3 = rhs(phi + 0.5 * h * k2)
    k4 = rhs(phi + h * k3)
    p = phi + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    p = jnp.clip(p, 0.0, None)
    return p / (p.sum() + 1e-30)


def _integrate_week(phi0, b, A):
    h = 1.0 / N_STEPS
    def scan_fn(phi, _):
        return _rk4_step(phi, b, A, h), None
    phi_final, _ = jax.lax.scan(scan_fn, phi0, None, length=N_STEPS)
    return phi_final


def _predict_patient(phi1, b, A):
    phi2 = _integrate_week(phi1, b, A)
    phi3 = _integrate_week(phi2, b, A)
    return phi2, phi3


def make_loss_jax(phi_obs_np):
    phi_obs = jnp.array(phi_obs_np)   # (n_p, 3, N_G)
    n_p = phi_obs.shape[0]

    def loss(theta):
        A     = theta[:N_A].reshape(N_G, N_G)
        b_all = theta[N_A:].reshape(n_p, N_G)
        sq = jnp.float64(0.0)
        for i in range(n_p):
            phi2_p, phi3_p = _predict_patient(phi_obs[i, 0], b_all[i], A)
            sq = sq + jnp.sum((phi2_p - phi_obs[i, 1])**2)
            sq = sq + jnp.sum((phi3_p - phi_obs[i, 2])**2)
        rmse = jnp.sqrt(sq / (n_p * 2 * N_G))
        reg  = LAMBDA_REG * jnp.sum(A**2)
        return rmse + reg

    return jax.jit(jax.value_and_grad(loss)), n_p


def make_bounds(n_p):
    a_bounds = [(None, 0.0) if i == j else (None, None)
                for i in range(N_G) for j in range(N_G)]
    return a_bounds + [(None, None)] * (n_p * N_G)


def run_one(x0, loss_and_grad, bounds, label=''):
    calls = [0]
    t0 = time.time()

    def scipy_fn(theta):
        val, grad = loss_and_grad(jnp.array(theta))
        calls[0] += 1
        if calls[0] % 20 == 0:
            print(f'  [{label}] iter {calls[0]:4d}  loss={float(val):.5f}'
                  f'  ({time.time()-t0:.0f}s)', flush=True)
        return float(val), np.array(grad, dtype=np.float64)

    res = minimize(scipy_fn, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                   options=dict(maxiter=2000, maxfun=50_000,
                                ftol=1e-12, gtol=1e-8, maxls=40))
    print(f'  [{label}] DONE loss={res.fun:.5f}  iters={calls[0]}'
          f'  {res.message[:60]}', flush=True)
    return res.fun, res.x


def save_result(A_np, b_np, loss_val, label, path):
    json.dump(dict(A=A_np.tolist(), b_all=b_np.tolist(),
                   rmse=float(loss_val), label=label,
                   guilds=GUILD_ORDER, patients=PATIENTS,
                   success=True, lambda_reg=LAMBDA_REG),
              open(path, 'w'), indent=2)


def main():
    print('Loading phi_guild...', flush=True)
    phi_obs = np.load(PHI_NPY)
    n_p = phi_obs.shape[0]
    print(f'  {n_p} patients, {phi_obs.shape[1]} weeks, {phi_obs.shape[2]} guilds',
          flush=True)

    print('Building JAX loss (JIT compile on first call ~10-30s)...', flush=True)
    loss_and_grad, _ = make_loss_jax(phi_obs)

    # JIT warmup
    _ = loss_and_grad(jnp.zeros(N_A + n_p * N_G))
    print('JIT compiled.', flush=True)

    bounds = make_bounds(n_p)

    # warm start from previous result
    prev_json = RESULTS_DIR / 'fit_guild.json'
    starts = []
    prev_rmse = np.inf
    if prev_json.exists():
        prev = json.load(open(prev_json))
        x_warm = pack(np.array(prev['A']), np.array(prev['b_all']))
        starts.append(('warm', x_warm))
        prev_rmse = prev['rmse']
        print(f'Warm start (prev RMSE={prev_rmse:.5f})', flush=True)

    # 3 random restarts
    rng = np.random.default_rng(42)
    for i in range(3):
        A_r = default_A() + rng.normal(0, 0.02, (N_G, N_G))
        np.fill_diagonal(A_r, np.minimum(A_r.diagonal(), 0))
        b_r = rng.uniform(0.05, 0.3, (n_p, N_G))
        starts.append((f'rand{i}', pack(A_r, b_r)))

    best_rmse, best_x = np.inf, None
    chk = RESULTS_DIR / 'fit_guild_jaxlbfgs_best.json'
    t_total = time.time()

    for label, x0 in starts:
        loss_val, x = run_one(x0, loss_and_grad, bounds, label=label)
        A_np = np.array(x[:N_A]).reshape(N_G, N_G)
        b_np = np.array(x[N_A:]).reshape(n_p, N_G)
        save_result(A_np, b_np, loss_val, label, chk)
        print(f'  Checkpoint -> {chk.name}', flush=True)
        if loss_val < best_rmse:
            best_rmse, best_x = loss_val, x

    print(f'\nBest loss: {best_rmse:.5f}  (elapsed {time.time()-t_total:.0f}s)',
          flush=True)

    A_map = np.array(best_x[:N_A]).reshape(N_G, N_G)
    b_map = np.array(best_x[N_A:]).reshape(n_p, N_G)

    out = RESULTS_DIR / 'fit_guild.json'
    if best_rmse < prev_rmse:
        save_result(A_map, b_map, best_rmse, 'JAX-LBFGS multi-start', out)
        print(f'Improved! fit_guild.json updated (RMSE {prev_rmse:.5f} → {best_rmse:.5f})',
              flush=True)
    else:
        print(f'No improvement over prev RMSE={prev_rmse:.5f}; fit_guild.json unchanged',
              flush=True)

    print('\nA matrix (rows=target, cols=source):', flush=True)
    print('         ' + ''.join(f'{g[:4]:>8s}' for g in GUILD_ORDER), flush=True)
    for i, row in enumerate(GUILD_ORDER):
        print(f'  {row[:8]:8s} ' + ''.join(f'{A_map[i,j]:8.3f}' for j in range(N_G)),
              flush=True)


if __name__ == '__main__':
    main()
