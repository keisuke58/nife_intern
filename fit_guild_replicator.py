#!/usr/bin/env python3
"""
Fit 10-guild replicator model to Dieckow phi_guild.npy.
Output: results/dieckow_cr/fit_guild.json
"""

import json, sys, time
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from guild_replicator_dieckow import (
    N_G, GUILD_ORDER, rmse_guild, default_A, pack, unpack,
)

RESULTS_DIR = Path(__file__).parent / 'results' / 'dieckow_cr'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PHI_NPY    = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'
LAMBDA_REG = 0.01
PATIENTS   = list('ABCDEFGHKL')

_call_count = [0]
_t0         = [0.0]


def make_loss(phi_obs):
    n_p = phi_obs.shape[0]
    def loss(theta):
        A, b_all = unpack(theta, n_p)
        r   = rmse_guild(A, b_all, phi_obs)
        reg = LAMBDA_REG * np.sum(A**2)
        val = r + reg
        _call_count[0] += 1
        if _call_count[0] % 20 == 0:
            elapsed = time.time() - _t0[0]
            print(f'  call {_call_count[0]:4d}  loss={val:.5f}  ({elapsed:.1f}s)', flush=True)
        return val
    return loss


def make_bounds(n_p):
    # A: diagonal ≤ 0 (self-limitation), off-diagonal unconstrained
    a_bounds = []
    for i in range(N_G):
        for j in range(N_G):
            a_bounds.append((None, 0.0) if i == j else (None, None))
    b_bounds = [(None, None)] * (n_p * N_G)
    return a_bounds + b_bounds


def run_one(x0, phi_obs, bounds, label=''):
    loss_fn = make_loss(phi_obs)
    result  = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=5000, maxfun=200_000,
                                    ftol=1e-12, gtol=1e-8, maxls=60))
    n_p = phi_obs.shape[0]
    A, b = unpack(result.x, n_p)
    rmse = rmse_guild(A, b, phi_obs)
    print(f'  [{label}] RMSE={rmse:.5f}  calls={_call_count[0]}  {result.message[:60]}',
          flush=True)
    return rmse, result.x


def main():
    print('Loading phi_guild...', flush=True)
    phi_obs = np.load(PHI_NPY)   # (10, 3, 10)
    n_p = phi_obs.shape[0]
    print(f'  {n_p} patients × {phi_obs.shape[1]} weeks × {phi_obs.shape[2]} guilds',
          flush=True)
    bounds = make_bounds(n_p)

    # --- warm start from previous best if available ---
    prev_json = RESULTS_DIR / 'fit_guild.json'
    if prev_json.exists():
        import json as _json
        prev = _json.load(open(prev_json))
        x_warm = pack(np.array(prev['A']), np.array(prev['b_all']))
        starts = [('warm', x_warm)]
        print('Using warm start from previous fit_guild.json', flush=True)
    else:
        starts = []

    # --- random restarts ---
    rng = np.random.default_rng(42)
    for i in range(4):
        A_rand = default_A() + rng.normal(0, 0.02, (N_G, N_G))
        np.fill_diagonal(A_rand, np.minimum(A_rand.diagonal(), 0))
        b_rand = rng.uniform(0.05, 0.3, (n_p, N_G))
        starts.append((f'rand{i}', pack(A_rand, b_rand)))

    _t0[0] = time.time()
    best_rmse, best_x = np.inf, None
    for label, x0 in starts:
        _call_count[0] = 0
        rmse, x = run_one(x0, phi_obs, bounds, label)
        if rmse < best_rmse:
            best_rmse, best_x = rmse, x

    print(f'\nBest RMSE: {best_rmse:.5f}  (total elapsed {time.time()-_t0[0]:.1f}s)',
          flush=True)
    A_map, b_map = unpack(best_x, n_p)
    rmse_final   = rmse_guild(A_map, b_map, phi_obs)
    result_msg   = f'Multi-start best (warm+4 random)'

    print(f'\nFinal RMSE: {rmse_final:.5f}', flush=True)
    print('\nEffective A matrix (rows=target, cols=source):', flush=True)
    header = '         ' + ''.join(f'{g[:4]:>8s}' for g in GUILD_ORDER)
    print(header, flush=True)
    for i, row_name in enumerate(GUILD_ORDER):
        vals = ''.join(f'{A_map[i,j]:8.3f}' for j in range(N_G))
        print(f'  {row_name[:8]:8s} {vals}', flush=True)

    out = RESULTS_DIR / 'fit_guild.json'
    json.dump(dict(
        A=A_map.tolist(), b_all=b_map.tolist(),
        rmse=rmse_final, guilds=GUILD_ORDER,
        patients=PATIENTS, success=True,
        message=result_msg, n_calls=_call_count[0],
        lambda_reg=LAMBDA_REG,
    ), open(out, 'w'), indent=2)
    print(f'\nSaved: {out}', flush=True)


if __name__ == '__main__':
    main()
