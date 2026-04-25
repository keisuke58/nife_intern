#!/usr/bin/env python3
"""
fit_guild_glv_struct.py — gLV + CLSM structural data integration.

Integrates PerLive into gLV by scaling patient-specific growth rates:
  b_p_eff(w) = b_p * (0.5 + 0.5 * q_{p,w})   [same α formula as Hamilton]

This lets us check: is the RMSE improvement from structural data
Hamilton-specific, or does it also help the simpler gLV?

Output: results/dieckow_cr/fit_guild_glv_struct.json
"""

import json, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import (
    GUILD_ORDER, N_G, default_A, pack, unpack,
)
from load_structure_dieckow import load_structural_data, build_occupancy

PHI_NPY   = _here / 'results' / 'dieckow_otu' / 'phi_guild_excel_class.npy'
WARM_JSON = _here / 'results' / 'dieckow_cr'  / 'fit_guild_excel_class.json'
OUT_JSON  = _here / 'results' / 'dieckow_cr'  / 'fit_guild_glv_struct.json'
STRUCT_XL = _here / 'Datasets' / 'Abutment_Structure vs composition.xlsx'

PATIENTS_ALL = list('ABCDEFGHKL')
DT = 168.0          # 1 week in hours (unused for dimensionless ODE; kept for label)
LAM  = 1e-4         # L2 on A off-diagonal
LAM_DIAG = 0.0      # diagonal unconstrained (kept ≤ 0 by post-processing)
N_STARTS = 5


def replicator_rhs(t, phi, b_eff, A):
    f = b_eff + A @ phi
    return phi * (f - phi @ f)


def integrate_step_struct(phi0, b_p, A, q_pw):
    """One week of gLV with PerLive-scaled growth rates."""
    b_eff = b_p * (0.5 + 0.5 * q_pw)
    sol = solve_ivp(replicator_rhs, [0, 1.0], phi0, args=(b_eff, A),
                    method='RK45', rtol=1e-6, atol=1e-8)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0


def rmse_struct(A, b_all, phi_obs, pl_arr):
    """RMSE with PerLive-scaled growth rates per patient per week."""
    n_p, n_w, n_g = phi_obs.shape
    sq_sum = 0.0; count = 0
    for p in range(n_p):
        phi2 = integrate_step_struct(phi_obs[p, 0], b_all[p], A, pl_arr[p, 0])
        phi3 = integrate_step_struct(phi2,           b_all[p], A, pl_arr[p, 1])
        sq_sum += np.sum((phi2 - phi_obs[p, 1])**2)
        sq_sum += np.sum((phi3 - phi_obs[p, 2])**2)
        count  += 2 * n_g
    return np.sqrt(sq_sum / count)


def objective(theta, phi_obs, pl_arr, n_g):
    n_p = phi_obs.shape[0]
    A = theta[:n_g * n_g].reshape(n_g, n_g)
    b_all = theta[n_g * n_g:].reshape(n_p, n_g)
    rmse = rmse_struct(A, b_all, phi_obs, pl_arr)
    reg  = LAM * np.sum(A**2)
    return rmse + reg


def make_bounds(n_p, n_g):
    # A diagonal ≤ 0, off-diagonal free; b free
    n_A = n_g * n_g
    bounds = []
    for i in range(n_g):
        for j in range(n_g):
            if i == j:
                bounds.append((None, 0.0))
            else:
                bounds.append((None, None))
    for _ in range(n_p * n_g):
        bounds.append((None, None))
    return bounds


def main():
    t0 = time.time()

    phi_all = np.load(PHI_NPY)
    n_p, n_w, n_g = phi_all.shape
    guilds = GUILD_ORDER[:n_g]
    present = phi_all.sum(axis=2) > 1e-12
    keep    = present[:, 0]
    phi_all  = phi_all[keep]
    patients = [p for k, p in zip(keep.tolist(), PATIENTS_ALL) if k]
    n_keep   = len(patients)

    print(f'Loaded: {n_keep} patients, {n_g} guilds', flush=True)

    # structural data
    struct  = load_structural_data(STRUCT_XL)
    occ_raw, _ = build_occupancy(struct, normalize=True)
    pl_raw  = struct.get('PerLive', {})
    pl_arr  = np.ones((n_keep, n_w))
    for p_idx, pat in enumerate(patients):
        for w in range(n_w):
            pl_arr[p_idx, w] = pl_raw.get((pat, w + 1), 100.0) / 100.0
    print(f'PerLive range: [{pl_arr.min():.3f}, {pl_arr.max():.3f}]', flush=True)

    # warm start
    A0 = default_A()[:n_g, :n_g]
    b0 = np.full((n_keep, n_g), 0.1)
    if Path(WARM_JSON).exists():
        d = json.load(open(WARM_JSON))
        A_w = np.array(d['A'])
        if A_w.shape[0] != n_g:
            A_w = A_w[:n_g, :n_g]
        A0 = A_w
        b_w = np.array(d['b_all'])
        b0  = b_w[:n_keep, :n_g] if b_w.shape[0] >= n_keep else np.full((n_keep, n_g), 0.1)
        print('Warm start: loaded fit_guild_excel_class.json', flush=True)

    rmse_init = rmse_struct(A0, b0, phi_all, pl_arr)
    print(f'Initial RMSE: {rmse_init:.5f}', flush=True)

    bounds = make_bounds(n_keep, n_g)
    best_rmse = float('inf')
    best_theta = None
    rng = np.random.default_rng(0)

    for s in range(N_STARTS):
        if s == 0:
            theta0 = np.concatenate([A0.ravel(), b0.ravel()])
        else:
            noise_A = rng.normal(0, 0.05, A0.shape)
            noise_b = rng.normal(0, 0.05, b0.shape)
            theta0  = np.concatenate([(A0 + noise_A).ravel(), (b0 + noise_b).ravel()])

        res = minimize(objective, theta0, args=(phi_all, pl_arr, n_g),
                       method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        A_opt = res.x[:n_g * n_g].reshape(n_g, n_g)
        b_opt = res.x[n_g * n_g:].reshape(n_keep, n_g)
        rmse_s = rmse_struct(A_opt, b_opt, phi_all, pl_arr)
        print(f'  start {s+1}/{N_STARTS}  RMSE={rmse_s:.5f}  ({time.time()-t0:.1f}s)', flush=True)
        if rmse_s < best_rmse:
            best_rmse = rmse_s; best_theta = res.x

    A_best = best_theta[:n_g * n_g].reshape(n_g, n_g)
    b_best = best_theta[n_g * n_g:].reshape(n_keep, n_g)
    print(f'\nBest RMSE (with struct): {best_rmse:.5f}', flush=True)

    # compare: plain gLV RMSE (PerLive=1 → no scaling)
    pl_ones  = np.ones_like(pl_arr)
    rmse_nostruct = rmse_struct(A_best, b_best, phi_all, pl_ones)
    print(f'Same A,b without struct scaling: {rmse_nostruct:.5f}', flush=True)

    json.dump(dict(
        A=A_best.tolist(),
        b_all=b_best.tolist(),
        rmse=best_rmse,
        rmse_no_struct=rmse_nostruct,
        guilds=guilds,
        patients=patients,
        pl_arr=pl_arr.tolist(),
        message=f'gLV+struct (PerLive-scaled b) L-BFGS-B lam={LAM} {N_STARTS} starts',
    ), open(OUT_JSON, 'w'), indent=2)
    print(f'Saved: {OUT_JSON}', flush=True)


if __name__ == '__main__':
    main()
