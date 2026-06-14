#!/usr/bin/env python3
"""
fit_dieckow_seed.py --seed N — Dieckow gLV no-prior fit from a given random init seed.
Saves the fitted A to results/duranpinedo_validation/A_dk_noprior_seed{N}.npy

Used to test whether the Dieckow vs Duran-Pinedo strong-pair sign agreement is an artifact
of both fits sharing the same default_rng(0) warm-start, or a genuine data signal.
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

ap = argparse.ArgumentParser(); ap.add_argument('--seed', type=int, required=True)
seed = ap.parse_args().seed

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G

DATA = _here / 'results' / 'dieckow_otu'
OUT  = _here / 'results' / 'duranpinedo_validation'
phi_all = np.load(DATA / 'phi_guild.npy')
N_P, N_T = phi_all.shape[:2]
LAM = 1e-4

def rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)

def integrate_step(phi0, b, A, dt=1.0):
    sol = solve_ivp(rhs, [0, dt], phi0, args=(b, A), method='RK45',
                    rtol=1e-6, atol=1e-8)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0.copy()

def train_obj(x):
    A = x[:N_G * N_G].reshape(N_G, N_G)
    b = x[N_G * N_G:].reshape(N_P, N_G)
    sq, cnt = 0.0, 0
    for p in range(N_P):
        phi = phi_all[p, 0].copy()
        for t in range(1, N_T):
            phi = integrate_step(phi, b[p], A)
            sq += np.sum((phi - phi_all[p, t]) ** 2); cnt += N_G
    return np.sqrt(sq / cnt) + LAM * np.sum(A ** 2)

bounds = [(None, 0.0) if i == j else (None, None)
          for i in range(N_G) for j in range(N_G)] + [(None, None)] * (N_P * N_G)

rng = np.random.default_rng(seed)
A0 = rng.normal(0, 0.01, (N_G, N_G)); np.fill_diagonal(A0, -0.1)
x0 = np.concatenate([A0.ravel(), np.zeros(N_P * N_G)])

t0 = time.time()
res = minimize(train_obj, x0, method='L-BFGS-B', bounds=bounds,
               options={'maxiter': 5000, 'maxfun': 40000, 'ftol': 1e-10, 'gtol': 1e-7})
A = res.x[:N_G * N_G].reshape(N_G, N_G)
np.save(OUT / f'A_dk_noprior_seed{seed}.npy', A)
print(f'seed={seed}  loss={res.fun:.5f}  t={time.time()-t0:.0f}s  → A_dk_noprior_seed{seed}.npy', flush=True)
