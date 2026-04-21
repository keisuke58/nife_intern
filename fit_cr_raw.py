#!/usr/bin/env python3
"""
Fit consumer-resource ODE to Dieckow raw 16S OTU data (phi_obs_raw.npy).
Unlike fit_cr_dieckow.py, b_all is initialised to 0.1 (no Hamilton prior).

Output: results/dieckow_cr/fit_cr_raw.json
"""

import json, sys, time
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from consumer_resource_dieckow import (
    N_SP, N_CR, LABELS,
    rmse_cr, effective_A_from_cr, default_theta_cr,
)

RESULTS_DIR = Path(__file__).parent / 'results' / 'dieckow_cr'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH  = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_obs_raw.npy'
LAMBDA_REG = 0.01
PATIENTS   = list('ABCDEFGHKL')


def pack(theta_cr, b_all):
    return np.concatenate([theta_cr, b_all.ravel()])

def unpack(theta_full, n_p):
    return theta_full[:N_CR], theta_full[N_CR:].reshape(n_p, N_SP)

def make_bounds(n_p):
    cr_bounds = [
        (0, None), (0, None), (0, None), (-5, 5),
        (0, None), (0, None), (0, None), (-5, 5),
        (0, None), (-5, 5),
        (0, None), (0, None), (0, None),
    ]
    return cr_bounds + [(None, None)] * (n_p * N_SP)

_call_count = [0]
_t0 = [0.0]

def make_loss(phi_obs):
    n_p = phi_obs.shape[0]
    def loss(theta_full):
        theta_cr, b_all = unpack(theta_full, n_p)
        r = rmse_cr(theta_cr, b_all, phi_obs)
        reg = LAMBDA_REG * np.sum(theta_cr**2)
        val = r + reg
        _call_count[0] += 1
        if _call_count[0] % 20 == 0:
            elapsed = time.time() - _t0[0]
            print(f"  call {_call_count[0]:4d}  loss={val:.5f}  ({elapsed:.1f}s)", flush=True)
        return val
    return loss


def main():
    print('Loading phi_obs_raw...', flush=True)
    phi_obs = np.load(DATA_PATH)   # (10, 3, 5)
    n_p = phi_obs.shape[0]
    print(f'  {n_p} patients × {phi_obs.shape[1]} weeks × {phi_obs.shape[2]} species', flush=True)

    # phi_obs here is (patients, weeks, species); rmse_cr expects (patients, timepoints, species)
    b_init = np.full((n_p, N_SP), 0.1)
    x0 = pack(default_theta_cr(), b_init)

    loss_fn = make_loss(phi_obs)
    bounds  = make_bounds(n_p)

    _t0[0] = time.time()
    print('Optimising with L-BFGS-B...', flush=True)
    result = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                      options=dict(maxiter=500, ftol=1e-9, gtol=1e-6, maxls=40))

    theta_cr_map, b_all_map = unpack(result.x, n_p)
    rmse_final = rmse_cr(theta_cr_map, b_all_map, phi_obs)
    A_eff = effective_A_from_cr(theta_cr_map).tolist()

    print(f'\nOptimisation: {result.message}', flush=True)
    print(f'Final RMSE: {rmse_final:.4f}  ({_call_count[0]} function evaluations)', flush=True)
    print('Effective A matrix:', flush=True)
    for i, row in enumerate(A_eff):
        print(f'  {LABELS[i]}: {[f"{v:.3f}" for v in row]}', flush=True)

    out = RESULTS_DIR / 'fit_cr_raw.json'
    json.dump(dict(
        theta_cr=theta_cr_map.tolist(), b_all=b_all_map.tolist(),
        rmse=rmse_final, A_effective=A_eff,
        patients=PATIENTS, success=result.success,
        message=result.message, n_calls=_call_count[0],
        lambda_reg=LAMBDA_REG,
    ), open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
