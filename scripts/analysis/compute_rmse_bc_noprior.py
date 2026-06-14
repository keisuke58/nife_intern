#!/usr/bin/env python3
"""
compute_rmse_bc_noprior.py — RMSE and Bray-Curtis (repo convention) for the
prior-free gLV fits, with the prior gLV Duran-Pinedo fit as a reference.

A is loaded from the saved fits; per-patient b is re-fit holding A fixed (at the
joint optimum b is already optimal given A, so this reproduces the in-sample fit
cheaply and recovers the b vectors that were not saved).

Prediction = rollout from the first observed timepoint (same scheme as
compute_bc_metrics.py).  RMSE/BC are computed per (patient, timepoint>0), averaged.
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G

OUT = _here / 'results' / 'duranpinedo_validation'

def bray_curtis(x, y):                       # repo convention (compute_bc_metrics.py:29)
    return 1.0 - 2.0 * np.minimum(x, y).sum() / (x.sum() + y.sum())

def rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)

def step(phi0, b, A):
    sol = solve_ivp(rhs, [0, 1.0], phi0, args=(b, A), method='RK45', rtol=1e-6, atol=1e-8)
    phi1 = np.clip(sol.y[:, -1], 0, None); s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0.copy()

def refit_b(phi, A):
    """optimize b_p (per patient) holding A fixed; returns b (N_P, N_G)."""
    N_P, N_T = phi.shape[:2]
    b = np.zeros((N_P, N_G))
    for p in range(N_P):
        def obj(bp):
            pr = phi[p, 0].copy(); sq = 0.0
            for t in range(1, N_T):
                pr = step(pr, bp, A)
                sq += np.sum((pr - phi[p, t]) ** 2)
            return sq
        r = minimize(obj, np.zeros(N_G), method='L-BFGS-B',
                     options={'maxiter': 2000, 'ftol': 1e-11})
        b[p] = r.x
    return b

def metrics(phi, A, label):
    N_P, N_T = phi.shape[:2]
    b = refit_b(phi, A)
    rmses, bcs = [], []
    for p in range(N_P):
        pr = phi[p, 0].copy()
        for t in range(1, N_T):
            pr = step(pr, b[p], A)
            rmses.append(float(np.sqrt(np.mean((pr - phi[p, t]) ** 2))))
            bcs.append(float(bray_curtis(pr, phi[p, t])))
    rmse_m, bc_m = float(np.mean(rmses)), float(np.mean(bcs))
    # pooled RMSE (sqrt of mean sq over all guild residuals) for continuity with earlier number
    print(f'  {label:28s}  RMSE={rmse_m:.4f}  BC={bc_m:.4f}   (n_pred={len(rmses)})')
    return {'label': label, 'rmse_mean': rmse_m, 'bc_mean': bc_m,
            'rmse_per_pred': rmses, 'bc_per_pred': bcs, 'n_pred': len(rmses)}

phi_dk  = np.load(_here / 'results' / 'dieckow_otu' / 'phi_guild.npy')   # (10,3,10)
phi_bot = np.load(_here / 'data' / 'prjna725874' / 'phi_guild.npy')      # (15,7,10)

A_dk_np  = np.array(json.load(open(OUT / 'dieckow_A_noprior_comparison.json'))['A_dieckow_noprior'])
A_bot_np = np.array(json.load(open(OUT / 'duranpinedo_A_comparison_noprior.json'))['A_duranpinedo'])
A_bot_pr = np.array(json.load(open(OUT / 'duranpinedo_A_comparison.json'))['A_duranpinedo'])

print('Training (in-sample) RMSE & Bray-Curtis — rollout from t0, repo BC convention\n')
print('Dieckow 2024 (10 pat × 3 wk):')
r1 = metrics(phi_dk,  A_dk_np,  'gLV no-prior')
print('\nDuran-Pinedo 2021 (15 pat × 7 tp):')
r2 = metrics(phi_bot, A_bot_np, 'gLV no-prior')
r3 = metrics(phi_bot, A_bot_pr, 'gLV AGORA prior (w=1.0)')

json.dump({'dieckow_noprior': r1, 'duranpinedo_noprior': r2, 'duranpinedo_prior_w1p0': r3},
          open(OUT / 'rmse_bc_noprior.json', 'w'), indent=2)
print(f'\nSaved → {OUT}/rmse_bc_noprior.json')
