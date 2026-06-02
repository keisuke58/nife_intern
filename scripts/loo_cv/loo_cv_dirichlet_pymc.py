"""
LOO-CV for Dirichlet-gLV (PyMC) on Dieckow 2024.
For each held-out patient:
  1. Fit A + b on 9 training patients (MAP via find_MAP)
  2. Fit b_ho for held-out patient (A fixed, wk1 data)
  3. Predict wk2, wk3; compute RMSE + Bray-Curtis

Usage:
  python loo_cv_dirichlet_pymc.py [--fold 0-9 | --all]
Output:
  results/dirichlet_loo/fold_{patient}.json
  results/dirichlet_loo/loo_summary.csv
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.spatial.distance import braycurtis
from scipy.integrate import solve_ivp

HERE = Path(__file__).resolve().parents[2]
OUT  = HERE / 'results/dirichlet_loo'
OUT.mkdir(parents=True, exist_ok=True)

# ── data ────────────────────────────────────────────────
with open(HERE / 'results/dieckow_otu/guild_summary.json') as f:
    gs = json.load(f)

GUILDS   = gs['guilds']
PATIENTS = list('ABCDEFGHKL')
N_G = len(GUILDS)
N_P = len(PATIENTS)

obs = {}
for s in gs['samples']:
    obs[(s['patient'], s['week'])] = np.array([s[g] for g in GUILDS], dtype=float)

def get_phi(patients):
    phi = np.zeros((len(patients), 3, N_G))
    for i, pat in enumerate(patients):
        for w in range(3):
            phi[i, w] = obs[(pat, w + 1)]
    phi = np.clip(phi, 1e-6, 1)
    phi /= phi.sum(axis=-1, keepdims=True)
    return phi

def replicator_step(phi, A, b, n_steps=20):
    dt = 1.0 / n_steps
    x = phi.copy()
    for _ in range(n_steps):
        f = b + A @ x
        fbar = x @ f
        x = x + dt * x * (f - fbar)
        x = np.clip(x, 1e-8, 1)
        x /= x.sum()
    return x

idx_u = np.triu_indices(N_G, k=1)

def build_training_model(phi_train, n_train):
    """Dirichlet-gLV model for n_train patients."""
    with pm.Model() as model:
        A_upper    = pm.Normal('A_upper', mu=0, sigma=0.5,
                               shape=N_G*(N_G-1)//2)
        A_diag_pos = pm.HalfNormal('A_diag_pos', sigma=0.5, shape=N_G)
        b_mat      = pm.Normal('b', mu=0, sigma=0.5, shape=(n_train, N_G))
        alpha      = pm.Exponential('alpha', lam=0.1)

        A = pt.zeros((N_G, N_G))
        A = pt.set_subtensor(A[idx_u[0], idx_u[1]], A_upper)
        A = pt.set_subtensor(A[idx_u[1], idx_u[0]], A_upper)
        A = pt.set_subtensor(A[np.arange(N_G), np.arange(N_G)],
                             -pt.abs(A_diag_pos))

        for i in range(n_train):
            b_p = b_mat[i]
            for w_from, w_to in [(0, 1), (1, 2)]:
                phi_in  = pt.as_tensor_variable(phi_train[i, w_from])
                phi_out = phi_train[i, w_to]
                f       = b_p + pt.dot(A, phi_in)
                fbar    = pt.dot(phi_in, f)
                dphi    = phi_in * (f - fbar)
                phi_pred = phi_in + dphi
                phi_pred = pt.clip(phi_pred, 1e-6, 1)
                phi_pred = phi_pred / phi_pred.sum()
                pm.Dirichlet(f'obs_{i}_w{w_to+1}',
                             a=alpha * phi_pred,
                             observed=phi_out)
    return model

def run_fold(hold_idx):
    pat_ho    = PATIENTS[hold_idx]
    pat_train = [p for i, p in enumerate(PATIENTS) if i != hold_idx]
    n_train   = len(pat_train)

    phi_train = get_phi(pat_train)
    phi_ho    = get_phi([pat_ho])[0]  # (3, N_G)

    print(f"\n=== Fold {hold_idx}: hold-out={pat_ho} ===")
    t0 = time.time()

    # Step 1: MAP on training patients
    model_train = build_training_model(phi_train, n_train)
    with model_train:
        map_est = pm.find_MAP(method='L-BFGS-B',
                              progressbar=False,
                              maxeval=2000)

    A_upper_map = map_est['A_upper']
    A_diag_map  = map_est['A_diag_pos']
    alpha_map   = float(map_est['alpha'])

    A_map = np.zeros((N_G, N_G))
    A_map[idx_u[0], idx_u[1]] = A_upper_map
    A_map[idx_u[1], idx_u[0]] = A_upper_map
    np.fill_diagonal(A_map, -np.abs(A_diag_map))

    # Step 2: fit b_ho from wk1 only (A fixed)
    with pm.Model() as model_ho:
        b_ho   = pm.Normal('b_ho', mu=0, sigma=0.5, shape=N_G)
        A_fix  = pt.as_tensor_variable(A_map)
        alpha_fix = pt.as_tensor_variable(alpha_map)

        phi_in  = pt.as_tensor_variable(phi_ho[0])
        phi_out = phi_ho[1]
        f       = b_ho + pt.dot(A_fix, phi_in)
        fbar    = pt.dot(phi_in, f)
        dphi    = phi_in * (f - fbar)
        phi_pred = phi_in + dphi
        phi_pred = pt.clip(phi_pred, 1e-6, 1)
        phi_pred = phi_pred / phi_pred.sum()
        pm.Dirichlet('obs_w2', a=alpha_fix * phi_pred, observed=phi_out)

        map_ho = pm.find_MAP(method='L-BFGS-B',
                             progressbar=False,
                             maxeval=500)

    b_ho_map = map_ho['b_ho']

    # Step 3: predict wk2, wk3
    phi_pred_w2 = replicator_step(phi_ho[0], A_map, b_ho_map)
    phi_pred_w3 = replicator_step(phi_ho[1], A_map, b_ho_map)

    rmse_w2 = float(np.sqrt(np.mean((phi_pred_w2 - phi_ho[1])**2)))
    rmse_w3 = float(np.sqrt(np.mean((phi_pred_w3 - phi_ho[2])**2)))
    rmse    = (rmse_w2 + rmse_w3) / 2

    bc_w2 = float(braycurtis(phi_pred_w2, phi_ho[1]))
    bc_w3 = float(braycurtis(phi_pred_w3, phi_ho[2]))
    bc    = (bc_w2 + bc_w3) / 2

    elapsed = time.time() - t0
    print(f"  RMSE={rmse:.4f}  BC={bc:.4f}  time={elapsed:.0f}s")

    result = dict(patient=pat_ho, hold_idx=hold_idx,
                  rmse=rmse, rmse_w2=rmse_w2, rmse_w3=rmse_w3,
                  bc=bc, bc_w2=bc_w2, bc_w3=bc_w3,
                  elapsed=elapsed,
                  A_upper=A_upper_map.tolist(),
                  A_diag=A_diag_map.tolist(),
                  b_ho=b_ho_map.tolist())

    with open(OUT / f'fold_{pat_ho}.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result

def summarise():
    results = []
    for pat in PATIENTS:
        fp = OUT / f'fold_{pat}.json'
        if fp.exists():
            with open(fp) as f:
                results.append(json.load(f))
    if not results:
        return
    df = pd.DataFrame(results)[['patient','rmse','bc','elapsed']]
    mean_row = pd.DataFrame([{'patient':'MEAN',
                               'rmse': df['rmse'].mean(),
                               'bc':   df['bc'].mean(),
                               'elapsed': df['elapsed'].sum()}])
    df = pd.concat([df, mean_row], ignore_index=True)
    df.to_csv(OUT / 'loo_summary.csv', index=False)
    print("\n=== LOO Summary ===")
    print(df.to_string(index=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None,
                        help='0-9 for single fold')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()

    if args.all:
        for i in range(N_P):
            run_fold(i)
        summarise()
    elif args.fold is not None:
        run_fold(args.fold)
        summarise()
    else:
        print("Use --fold N (0-9) or --all")
