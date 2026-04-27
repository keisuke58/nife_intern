#!/usr/bin/env python3
"""
loo_hamilton_approx.py — Approximate LOO-CV for Hamilton+struct.

Strategy: fix A from full-cohort fit (fit_guild_hamilton_masked.json),
re-fit patient-specific b from W1→W2 only, predict W3.
This avoids re-fitting A (66 params) and runs ~6× faster.

Output: results/dieckow_cr/loo_cv_hamilton.json
"""

import json, os, sys, time
from pathlib import Path

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

import numpy as np
from scipy.optimize import minimize

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from load_structure_dieckow import load_structural_data, build_occupancy

import jax, jax.numpy as jnp
jax.config.update('jax_enable_x64', True)

sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
from hamilton_ode_jax_nsp import simulate_0d_nsp

CR_DIR  = _here / 'results' / 'dieckow_cr'
OTU_DIR = _here / 'results' / 'dieckow_otu'
STRUCT_XL = _here / 'Datasets' / 'Abutment_Structure vs composition.xlsx'
OUT     = CR_DIR / 'loo_cv_hamilton.json'
PAT_ALL = list('ABCDEFGHKL')

# ── Load data ─────────────────────────────────────────────────────���────────────
warm = json.load(open(CR_DIR / 'fit_guild_hamilton_masked.json'))
A_full = np.array(warm['A'])
b_full = np.array(warm['b_all'])
patients = warm['patients']          # e.g. 8 patients
n_p = len(patients)
n_sp = A_full.shape[0]

PHI_NPY = OTU_DIR / 'phi_guild_excel_class.npy'
phi_raw = np.load(PHI_NPY)           # (10, 3, n_g_raw)
pat_idx = [PAT_ALL.index(p) for p in patients]
phi_all = phi_raw[pat_idx][:, :, :n_sp].astype(float)

struct  = load_structural_data(STRUCT_XL)
occ_raw, _ = build_occupancy(struct, normalize=True)
pl_raw  = struct.get('PerLive', {})
pl_arr  = np.ones((n_p, 3))
occ_norm = np.ones((n_p, 3))
for pi, pat in enumerate(patients):
    for w in range(3):
        pl_arr[pi, w]   = pl_raw.get((pat, w+1), 100.0) / 100.0
        occ_norm[pi, w] = occ_raw.get((pat, w+1), 1.0)

n_A = n_sp * (n_sp + 1) // 2
N_STEPS = 2500

def pack_upper(A):
    v = []
    for j in range(n_sp):
        for i in range(j+1):
            v.append(A[i, j])
    return np.array(v)

def unpack_upper(v):
    A = np.zeros((n_sp, n_sp))
    idx = 0
    for j in range(n_sp):
        for i in range(j+1):
            A[i, j] = A[j, i] = v[idx]; idx += 1
    return A

@jax.jit
def _sim_jit(theta, phi_init, psi_val, alpha_val):
    phibar = simulate_0d_nsp(theta, n_sp=n_sp, n_steps=N_STEPS, dt=1e-4,
                             phi_init=phi_init, psi_init=psi_val,
                             c_const=25.0, alpha_const=alpha_val)
    eq = phibar[-1]; s = eq.sum()
    return jnp.where(s > 1e-10, eq / s, jnp.ones(n_sp) / n_sp)

print('Compiling Hamilton forward pass...', flush=True)
A_up_full = pack_upper(A_full)
_dummy_theta = np.concatenate([A_up_full, b_full[0]])
_ = np.array(_sim_jit(jnp.array(_dummy_theta), jnp.ones(n_sp)/n_sp,
                      jnp.array(0.5), jnp.array(50.0)))
print('Compiled.', flush=True)

def sim_np(A_up, b_p, phi0, psi, alpha):
    theta = np.concatenate([A_up, b_p])
    return np.array(_sim_jit(jnp.array(theta), jnp.array(phi0),
                             jnp.array(float(psi)), jnp.array(float(alpha))))

LAM_B = 1e-3

def loss_b(b, A_up, phi_ph, occ_ph, pl_ph):
    phi_w1 = phi_ph[0] * occ_ph[0]
    pred_w2 = sim_np(A_up, b, phi_w1,
                     float(np.clip(pl_ph[0], 1e-4, 0.9999)),
                     100.0 * (0.5 + 0.5 * pl_ph[0]))
    return float(np.mean((pred_w2 - phi_ph[1])**2) + LAM_B * np.sum(b**2))

# ── Approximate LOO ────────────────────────────────────────────────────────────
print(f'=== Approximate LOO-CV Hamilton+struct ({n_p} patients) ===', flush=True)
loo_rmses, results = [], []

for p_held in range(n_p):
    t0 = time.time()
    phi_ph  = phi_all[p_held]
    occ_ph  = occ_norm[p_held]
    pl_ph   = pl_arr[p_held]

    b0 = b_full[p_held] if p_held < len(b_full) else np.full(n_sp, 0.1)
    res = minimize(loss_b, b0, args=(A_up_full, phi_ph, occ_ph, pl_ph),
                   method='L-BFGS-B',
                   options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6})
    b_fit = res.x

    phi_w1 = phi_ph[0] * occ_ph[0]
    pred_w2 = sim_np(A_up_full, b_fit, phi_w1,
                     float(np.clip(pl_ph[0], 1e-4, 0.9999)),
                     100.0 * (0.5 + 0.5 * pl_ph[0]))
    phi_w2_abs = pred_w2 * occ_ph[1]
    pred_w3 = sim_np(A_up_full, b_fit, phi_w2_abs,
                     float(np.clip(pl_ph[1], 1e-4, 0.9999)),
                     100.0 * (0.5 + 0.5 * pl_ph[1]))
    rmse_p = float(np.sqrt(
        (np.sum((pred_w2 - phi_ph[1])**2) + np.sum((pred_w3 - phi_ph[2])**2))
        / (2 * n_sp)))
    loo_rmses.append(rmse_p)
    results.append({'patient': patients[p_held], 'rmse': rmse_p})
    print(f'  {patients[p_held]}: RMSE={rmse_p:.5f}  ({time.time()-t0:.1f}s)', flush=True)

loo_mean = float(np.mean(loo_rmses))
print(f'\nLOO mean RMSE (Hamilton+struct approx): {loo_mean:.5f}', flush=True)

json.dump({'loo_rmse_mean': loo_mean, 'per_patient': results,
           'model': 'Hamilton+struct LOO-CV (approx: A fixed, b refitted)'},
          open(OUT, 'w'), indent=2)
print(f'Saved: {OUT}', flush=True)
