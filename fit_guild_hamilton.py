#!/usr/bin/env python3
"""
Class-level Hamilton ODE fit to Dieckow phi_guild.npy.

Uses simulate_0d_nsp (n_sp=10) for the forward pass.
Equilibrium phibar[-1] → normalise → weekly composition prediction.

Symmetric A (N(N+1)/2 upper-triangle params, shared across patients)
Per-patient b (N guilds × 10 patients)
Total: N(N+1)/2 + 10N params

Optimisation: scipy L-BFGS-B with JIT forward pass + numerical gradients.
NOTE: JIT compile for n_sp=10 takes ~10-20 min. Submit via PBS.

Output: results/dieckow_cr/fit_guild_hamilton.json
"""

import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

from guild_replicator_dieckow import GUILD_ORDER, N_G

sys.path.insert(0, '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/main')
from hamilton_ode_jax_nsp import simulate_0d_nsp

PHI_NPY     = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'
RESULTS_DIR = Path(__file__).parent / 'results' / 'dieckow_cr'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SP    = N_G
N_A     = N_SP * (N_SP + 1) // 2
N_STEPS = 300                        # steps to equilibrium (0.25 time units)
LAMBDA  = 1e-2
PATIENTS = list('ABCDEFGHKL')


# ── JIT-compiled single-patient equilibrium prediction ────────────────────────

@jax.jit
def _eq_phi_jax(A_upper, b, phi_init):
    """Return normalised equilibrium phibar for one patient/week."""
    theta = jnp.concatenate([A_upper, b])
    phibar = simulate_0d_nsp(theta, n_sp=N_SP, n_steps=N_STEPS,
                              dt=1e-4, phi_init=phi_init,
                              c_const=25.0, alpha_const=100.0)
    eq = phibar[-1]
    s  = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(N_SP) / N_SP)


def eq_phi(A_upper_np, b_np, phi_init_np):
    """NumPy wrapper around JIT function."""
    return np.array(_eq_phi_jax(
        jnp.array(A_upper_np), jnp.array(b_np), jnp.array(phi_init_np)
    ))


# ── Pack / unpack ─────────────────────────────────────────────────────────────

def pack(A_upper, b_all):
    return np.concatenate([A_upper, b_all.ravel()])


def unpack(x, n_p):
    A_upper = x[:N_A]
    b_all   = x[N_A:].reshape(n_p, N_SP)
    return A_upper, b_all


def default_A_upper():
    """Diagonal -0.1, off-diagonal 0 (self-limitation only)."""
    A = -0.1 * np.eye(N_SP)
    upper = []
    for j in range(N_SP):
        for i in range(j + 1):
            upper.append(A[i, j])
    return np.array(upper)


# ── RMSE ─────────────────────────────────────────────────────────────────────

def rmse_hamilton(A_upper, b_all, phi_obs):
    n_p = phi_obs.shape[0]
    sq  = 0.0
    for p in range(n_p):
        phi_W1 = phi_obs[p, 0]
        b_p    = b_all[p]
        phi2   = eq_phi(A_upper, b_p, phi_W1)
        phi3   = eq_phi(A_upper, b_p, phi2)
        sq += np.sum((phi2 - phi_obs[p, 1]) ** 2)
        sq += np.sum((phi3 - phi_obs[p, 2]) ** 2)
    return np.sqrt(sq / (n_p * 2 * N_SP))


# ── Optimisation ─────────────────────────────────────────────────────────────

_call_count = [0]
_t0         = [0.0]


def make_loss(phi_obs):
    n_p = phi_obs.shape[0]

    def loss(x):
        A_upper, b_all = unpack(x, n_p)
        # upper-triangle parameterises symmetric A; get full for reg
        A_full = np.zeros((N_SP, N_SP))
        idx = 0
        for j in range(N_SP):
            for i in range(j + 1):
                A_full[i, j] = A_upper[idx]
                A_full[j, i] = A_upper[idx]
                idx += 1
        r   = rmse_hamilton(A_upper, b_all, phi_obs)
        reg = LAMBDA * np.sum(A_full ** 2)
        val = r + reg
        _call_count[0] += 1
        if _call_count[0] % 5 == 0:
            print(f'  call {_call_count[0]:4d}  loss={val:.5f}  rmse={r:.5f}'
                  f'  ({time.time()-_t0[0]:.1f}s)', flush=True)
        return float(val)

    return loss


def make_bounds(n_p):
    # A_upper: diagonal entries (A_ii) ≤ 0; off-diag unconstrained
    diag_positions = set()
    idx = 0
    for j in range(N_SP):
        for i in range(j + 1):
            if i == j:
                diag_positions.add(idx)
            idx += 1
    a_bounds = [(None, 0.0) if k in diag_positions else (None, None)
                for k in range(N_A)]
    b_bounds = [(None, None)] * (n_p * N_SP)
    return a_bounds + b_bounds


def main():
    print('Loading phi_guild...', flush=True)
    phi_obs = np.load(PHI_NPY)
    n_p = phi_obs.shape[0]
    print(f'  {n_p} patients × {phi_obs.shape[1]} weeks × {phi_obs.shape[2]} guilds',
          flush=True)

    # Warm JIT compile
    print('\nWarm-up JIT compile (this may take 10-20 min for n_sp=10)...', flush=True)
    t_compile = time.time()
    _ = eq_phi(default_A_upper(), np.full(N_SP, 0.1), phi_obs[0, 0])
    print(f'JIT compile done in {time.time()-t_compile:.1f}s', flush=True)

    bounds = make_bounds(n_p)

    # Warm start from gLV result if available (map asymmetric A → upper triangle)
    prev = RESULTS_DIR / 'fit_guild.json'
    starts = []
    if prev.exists():
        d     = json.load(open(prev))
        A_glv = np.array(d['A'])
        # Symmetrise gLV A for Hamilton warm start
        A_sym = (A_glv + A_glv.T) / 2.0
        upper_warm = []
        for j in range(N_SP):
            for i in range(j + 1):
                upper_warm.append(A_sym[i, j])
        b_warm = np.array(d['b_all'])
        starts.append(('warm_glv', pack(np.array(upper_warm), b_warm)))
        print('Warm start from fit_guild.json (gLV → symmetrised A)', flush=True)

    rng = np.random.default_rng(42)
    A_rand  = default_A_upper() + rng.normal(0, 0.02, N_A)
    b_rand  = rng.uniform(0.05, 0.3, (n_p, N_SP))
    starts.append(('rand0', pack(A_rand, b_rand)))

    loss_fn = make_loss(phi_obs)
    _t0[0]  = time.time()

    best_rmse, best_x = np.inf, None
    for label, x0 in starts:
        _call_count[0] = 0
        print(f'\n── Start: {label} ──', flush=True)
        res = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=2000, maxfun=500,
                                    ftol=1e-10, gtol=1e-7, maxls=30))
        A_upper, b_all = unpack(res.x, n_p)
        rmse = rmse_hamilton(A_upper, b_all, phi_obs)
        print(f'  [{label}] RMSE={rmse:.5f}  calls={_call_count[0]}'
              f'  {res.message[:60]}', flush=True)
        if rmse < best_rmse:
            best_rmse, best_x = rmse, res.x.copy()

    A_upper, b_all = unpack(best_x, n_p)
    rmse_final = rmse_hamilton(A_upper, b_all, phi_obs)

    # Reconstruct full symmetric A for output
    A_full = np.zeros((N_SP, N_SP))
    idx = 0
    for j in range(N_SP):
        for i in range(j + 1):
            A_full[i, j] = A_upper[idx]
            A_full[j, i] = A_upper[idx]
            idx += 1

    print(f'\nFinal RMSE: {rmse_final:.5f}  ({time.time()-_t0[0]:.1f}s total)', flush=True)
    print('\nEffective A matrix (rows=target, cols=source):', flush=True)
    header = '         ' + ''.join(f'{g[:4]:>8s}' for g in GUILD_ORDER)
    print(header, flush=True)
    for i, name in enumerate(GUILD_ORDER):
        row = ''.join(f'{A_full[i,j]:8.3f}' for j in range(N_SP))
        print(f'  {name[:8]:8s} {row}', flush=True)

    out = RESULTS_DIR / 'fit_guild_hamilton.json'
    json.dump(dict(
        A=A_full.tolist(), A_upper=A_upper.tolist(),
        b_all=b_all.tolist(), rmse=rmse_final,
        guilds=GUILD_ORDER, patients=PATIENTS,
        n_steps=N_STEPS, lambda_reg=LAMBDA,
        message=f'Hamilton ODE n_steps={N_STEPS} L-BFGS-B',
        n_calls=_call_count[0],
    ), open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
