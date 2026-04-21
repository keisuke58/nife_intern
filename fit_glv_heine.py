#!/usr/bin/env python3
"""
gLV replicator ODE fit to Heine 2025 5-species biofilm data.

Per condition (CS/CH/DS/DH): fit A (5x5 asymmetric) + b (5) to minimise
RMSE of integrated trajectory vs median observed composition.

ODE: dphi_i/dt = phi_i * (sum_j A_ij phi_j + b_i - fbar)
Initial cond: day 1 median, targets: days 3,6,10,15,21.

Output: results/heine2025/fit_glv_heine.json
"""

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

DATA_CSV    = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data/fig3_species_distribution_replicates.csv')
RESULTS_DIR = Path('/home/nishioka/IKM_Hiwi/nife/results/heine2025')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DAYS   = [1, 3, 6, 10, 15, 21]
N_SP   = 5
LAMBDA = 1e-3

CONDITIONS = [
    ('Commensal', 'Static',  'CS'),
    ('Commensal', 'HOBIC',   'CH'),
    ('Dysbiotic', 'Static',  'DS'),
    ('Dysbiotic', 'HOBIC',   'DH'),
]
SPECIES = {
    'Commensal': ['S. oralis', 'A. naeslundii', 'V. dispar',  'F. nucleatum', 'P. gingivalis_20709'],
    'Dysbiotic': ['S. oralis', 'A. naeslundii', 'V. parvula', 'F. nucleatum', 'P. gingivalis_W83'],
}
SHORT = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']


def replicator_rhs(t, phi, A, b):
    phi = np.maximum(phi, 1e-10)
    phi = phi / phi.sum()
    f    = A @ phi + b
    fbar = phi @ f
    return phi * (f - fbar)


def integrate_glv(A, b, phi0, days):
    sol = solve_ivp(replicator_rhs, [days[0], days[-1]], phi0,
                    t_eval=days, args=(A, b), method='RK45',
                    rtol=1e-6, atol=1e-9, max_step=1.0)
    traj = sol.y.T
    traj = np.maximum(traj, 0)
    traj = traj / traj.sum(axis=1, keepdims=True)
    return traj


def rmse_traj(A, b, phi_obs):
    try:
        pred = integrate_glv(A, b, phi_obs[0], DAYS)
        return float(np.sqrt(np.mean((pred - phi_obs) ** 2)))
    except Exception:
        return 1.0


def make_loss(phi_obs):
    def loss(x):
        A = x[:N_SP * N_SP].reshape(N_SP, N_SP)
        b = x[N_SP * N_SP:]
        r   = rmse_traj(A, b, phi_obs)
        reg = LAMBDA * np.sum(A ** 2)
        return r + reg
    return loss


def load_phi(df, condition, cultivation):
    sp_list = SPECIES[condition]
    mask    = (df['condition'] == condition) & (df['cultivation'] == cultivation)
    sub     = df[mask]
    phi     = np.zeros((len(DAYS), N_SP))
    for j, sp in enumerate(sp_list):
        sp_sub = sub[sub['species'] == sp]
        for i, day in enumerate(DAYS):
            vals = sp_sub[sp_sub['day'] == day]['distribution_pct'].values
            phi[i, j] = np.median(vals) if len(vals) > 0 else 0.0
    row_sums = phi.sum(axis=1, keepdims=True)
    phi = phi / np.where(row_sums > 0, row_sums, 1.0)
    return phi


def fit_condition(phi_obs, rng):
    loss_fn = make_loss(phi_obs)
    best_val, best_x = np.inf, None
    n = N_SP * N_SP + N_SP

    starts = [
        np.concatenate([-0.1 * np.eye(N_SP).ravel(), np.full(N_SP, 0.1)]),
        *[rng.normal(0, 0.15, n) for _ in range(4)],
    ]

    for x0 in starts:
        res = minimize(loss_fn, x0, method='L-BFGS-B',
                       options=dict(maxiter=3000, maxfun=10000, ftol=1e-12, gtol=1e-8))
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x.copy()

    A    = best_x[:N_SP * N_SP].reshape(N_SP, N_SP)
    b    = best_x[N_SP * N_SP:]
    rmse = rmse_traj(A, b, phi_obs)
    return A, b, rmse


def main():
    df  = pd.read_csv(DATA_CSV)
    rng = np.random.default_rng(42)
    t0  = time.time()
    print(f'Loaded {len(df)} rows\n')

    results = {}
    for condition, cultivation, label in CONDITIONS:
        phi_obs = load_phi(df, condition, cultivation)
        print(f'── {label} ──  phi_obs: {phi_obs.shape}')
        print(f'  day1: {" ".join(f"{v:.2f}" for v in phi_obs[0])}')

        A, b, rmse = fit_condition(phi_obs, rng)
        print(f'  RMSE={rmse:.5f}  ({time.time()-t0:.1f}s)')

        # Print A
        header = '       ' + ''.join(f'{s:>8s}' for s in SHORT)
        print(f'  {header}')
        for i, s in enumerate(SHORT):
            row = ''.join(f'{A[i,j]:8.3f}' for j in range(N_SP))
            print(f'  {s:6s} {row}')

        results[label] = dict(
            A=A.tolist(), b=b.tolist(), rmse=rmse,
            species=SPECIES[condition],
            condition=condition, cultivation=cultivation,
        )

    out = RESULTS_DIR / 'fit_glv_heine.json'
    json.dump(results, open(out, 'w'), indent=2)
    print(f'\nSaved: {out}')
    print(f'Total: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
