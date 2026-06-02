#!/usr/bin/env python3
"""
loo_nsp_full_gpu.py — Full LOO-CV for NSP Hamilton ODE on GPU.

Uses JAX JIT for forward pass only (no backward compile).
Gradients via scipy L-BFGS-B numerical finite differences.

Strategy per fold:
  1. Optimize A (55 params) with b from warm-start, L-BFGS-B
  2. Optimize b_train (n_train×10) with A fixed, L-BFGS-B
  3. Refit b_test (10 params) from W1→W2, L-BFGS-B
  4. Predict W3

Avoids JAX autograd compilation entirely.

Usage (vancouver01):
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate gnnEmulEnv
  CUDA_VISIBLE_DEVICES=0 python3 loo_nsp_full_gpu.py
  python3 loo_nsp_full_gpu.py --n-steps 100 --maxfun 2000
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse, json, sys, time
import os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TMPDIR', str(Path.home() / 'tmp'))

import jax
import jax.numpy as jnp
from jax import jit, vmap

jax.config.update('jax_enable_x64', True)

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from guild_replicator_dieckow import GUILD_ORDER
from hamilton_ode_jax_nsp import simulate_0d_nsp

N_SP    = len(GUILD_ORDER)
N_A     = N_SP * (N_SP + 1) // 2    # 55
PAT_ALL = list('ABCDEFGHKL')

# Diagonal positions in upper-triangle layout
_DIAG_IDX = [j * (j + 1) // 2 + j for j in range(N_SP)]


def default_A_upper():
    A = -0.1 * np.eye(N_SP)
    return np.array([A[i, j] for j in range(N_SP) for i in range(j + 1)])


def A_upper_to_full(A_upper):
    A = np.zeros((N_SP, N_SP))
    idx = 0
    for j in range(N_SP):
        for i in range(j + 1):
            A[i, j] = A_upper[idx]
            A[j, i] = A_upper[idx]
            idx += 1
    return A


def full_to_A_upper(A):
    return np.array([A[i, j] for j in range(N_SP) for i in range(j + 1)])


# ── Batched JIT forward (vmap over patients, forward-only — no grad needed) ───

def make_batch_fns(n_steps, dt=0.01):
    """Return batched (vmap) forward functions. Only forward, never grad."""

    def _single(A_upper, b_p, phi_init):
        theta  = jnp.concatenate([A_upper, b_p, jnp.zeros(1)])
        phibar = simulate_0d_nsp(
            theta, n_sp=N_SP, n_steps=n_steps, dt=dt,
            phi_init=phi_init, psi_init=0.999,
            c_const=25.0, alpha_const=100.0,
        )
        eq = phibar[-1]; s = eq.sum()
        return jnp.where(s > 1e-10, eq / s, jnp.ones(N_SP) / N_SP)

    # vmap over patients (axis 0 of b_all and phi_init_all)
    _batch = jit(vmap(_single, in_axes=(None, 0, 0)))

    def batch_np(A_upper_np, b_all_np, phi_init_all_np):
        """Compute all patients at once. Returns (n_patients, N_SP) ndarray."""
        return np.array(_batch(
            jnp.array(A_upper_np,    dtype=jnp.float64),
            jnp.array(b_all_np,      dtype=jnp.float64),
            jnp.array(phi_init_all_np, dtype=jnp.float64),
        ))

    # Single-patient wrapper for test patient refit
    _single_jit = jit(_single)
    def single_np(A_upper_np, b_np, phi_init_np):
        return np.array(_single_jit(
            jnp.array(A_upper_np,  dtype=jnp.float64),
            jnp.array(b_np,        dtype=jnp.float64),
            jnp.array(phi_init_np, dtype=jnp.float64),
        ))

    return batch_np, single_np


# ── Loss functions (NumPy, batched JAX forward) ───────────────────────────────

def _clip_diag(A_upper):
    A_upper = np.array(A_upper)
    for idx in _DIAG_IDX:
        A_upper[idx] = min(A_upper[idx], 0.0)
    return A_upper


def loss_A(A_upper, b_all, phi_train_np, mask_train_np, batch_np, lam_A):
    """Loss over training patients, optimising A only (b fixed)."""
    A_upper = _clip_diag(A_upper)
    phi_W1  = phi_train_np[:, 0, :]          # (n_train, N_SP)
    phi_W2o = phi_train_np[:, 1, :]
    phi_W3o = phi_train_np[:, 2, :]
    m2 = mask_train_np[:, 1]; m3 = mask_train_np[:, 2]

    pw2 = batch_np(A_upper, b_all, phi_W1)   # (n_train, N_SP)
    pw3 = batch_np(A_upper, b_all, pw2)

    sq  = np.sum(m2[:, None] * (pw2 - phi_W2o)**2)
    sq += np.sum(m3[:, None] * (pw3 - phi_W3o)**2)
    cnt = np.sum(m2 + m3) * N_SP
    rmse = float(np.sqrt(sq / cnt)) if cnt > 0 else 0.0
    return rmse + lam_A * float(np.sum(A_upper**2))


def loss_b_all(b_flat, A_upper, phi_train_np, mask_train_np, batch_np, lam_b, n_train):
    """Loss over training patients, optimising b_train (A fixed)."""
    b_all   = b_flat.reshape(n_train, N_SP)
    phi_W1  = phi_train_np[:, 0, :]
    phi_W2o = phi_train_np[:, 1, :]
    phi_W3o = phi_train_np[:, 2, :]
    m2 = mask_train_np[:, 1]; m3 = mask_train_np[:, 2]

    pw2 = batch_np(A_upper, b_all, phi_W1)
    pw3 = batch_np(A_upper, b_all, pw2)

    sq  = np.sum(m2[:, None] * (pw2 - phi_W2o)**2)
    sq += np.sum(m3[:, None] * (pw3 - phi_W3o)**2)
    cnt = np.sum(m2 + m3) * N_SP
    rmse = float(np.sqrt(sq / cnt)) if cnt > 0 else 0.0
    return rmse + lam_b * float(np.sum(b_flat**2))


def loss_b_test(b_p, A_upper, phi_test_np, mask_test_np, single_np, lam_b):
    """W1→W2 loss for one test patient."""
    pw2 = single_np(A_upper, b_p, phi_test_np[0])
    sq  = mask_test_np[1] * np.sum((pw2 - phi_test_np[1])**2)
    cnt = mask_test_np[1] * N_SP
    rmse = float(np.sqrt(sq / cnt)) if cnt > 0 else 0.0
    return rmse + lam_b * float(np.sum(b_p**2))


# ── LOO-CV ────────────────────────────────────────────────────────────────────

def run_loo(phi_all, patients, args):
    n_p   = len(patients)
    mask  = (phi_all.sum(axis=2) > 1e-12).astype(np.float64)
    lam_A = args.lambda_a
    lam_b = args.lambda_b

    # JIT compile batched forward (vmap over patients, forward-only — fast compile)
    print('\nJIT compiling batched forward (vmap, forward-only)...', flush=True)
    t0 = time.time()
    batch_np, single_np = make_batch_fns(args.n_steps, dt=args.dt)
    _ = batch_np(default_A_upper(), np.full((n_p, N_SP), 0.1), phi_all[:, 0, :])
    print(f'  batch forward compiled: {time.time()-t0:.1f}s  ({n_p} patients)', flush=True)
    _ = single_np(default_A_upper(), np.full(N_SP, 0.1), phi_all[0, 0])
    print(f'  single forward compiled: {time.time()-t0:.1f}s', flush=True)

    # Warm start
    warm_paths = [
        _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked_v2.json',
        _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json',
        _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton.json',
    ]
    warm_A = None; warm_b_map = {}
    for wp in warm_paths:
        if not wp.exists(): continue
        d  = json.load(open(wp))
        A0 = np.array(d['A'])
        if A0.shape[0] != N_SP: continue
        A_sym    = (A0 + A0.T) / 2.0
        warm_A   = full_to_A_upper(A_sym)
        stored   = d.get('patients', [])
        b0       = np.array(d['b_all'])
        if b0.shape[1] < N_SP:
            b0 = np.pad(b0, ((0, 0), (0, N_SP - b0.shape[1])))
        warm_b_map = {p: b0[i] for i, p in enumerate(stored)}
        print(f'Warm start from {wp.name}', flush=True)
        break
    if warm_A is None:
        warm_A = default_A_upper()

    # Diagonal-constraint bounds for A
    a_bounds = [(-np.inf, 0.0) if k in _DIAG_IDX else (-np.inf, np.inf)
                for k in range(N_A)]

    results = []

    for fold, test_pat in enumerate(patients):
        print(f'\n{"="*60}', flush=True)
        print(f'Fold {fold+1}/{n_p}  leave-out: {test_pat}', flush=True)
        t_fold = time.time()

        train_idx = [i for i, p in enumerate(patients) if p != test_pat]
        test_idx  = patients.index(test_pat)
        n_train   = len(train_idx)

        phi_train_np  = phi_all[np.array(train_idx)]   # (n_train, 3, N_SP)
        mask_train_np = mask[np.array(train_idx)]       # (n_train, 3)
        phi_test_np   = phi_all[test_idx]               # (3, N_SP)
        mask_test_np  = mask[test_idx]                  # (3,)

        # Init warm-start b for training patients
        b_warm = np.array([warm_b_map.get(patients[i], np.full(N_SP, 0.1))
                           for i in train_idx])

        # ── Step 1: Optimise A (b fixed at warm start) ───────────────────
        res_A = minimize(
            loss_A,
            x0=warm_A.copy(),
            args=(b_warm, phi_train_np, mask_train_np, batch_np, lam_A),
            method='L-BFGS-B',
            bounds=a_bounds,
            options=dict(maxiter=args.maxiter, maxfun=args.maxfun,
                         ftol=1e-10, gtol=1e-7),
        )
        A_opt = _clip_diag(res_A.x)
        print(f'  Step1 A: loss={res_A.fun:.5f}  iters={res_A.nit}'
              f'  ({time.time()-t_fold:.1f}s)', flush=True)

        # ── Step 2: Optimise b_train (A fixed) ──────────────────────────
        b_bounds = [(-np.inf, np.inf)] * (n_train * N_SP)
        res_b = minimize(
            loss_b_all,
            x0=b_warm.ravel(),
            args=(A_opt, phi_train_np, mask_train_np, batch_np, lam_b, n_train),
            method='L-BFGS-B',
            bounds=b_bounds,
            options=dict(maxiter=args.maxiter, maxfun=args.maxfun,
                         ftol=1e-10, gtol=1e-7),
        )
        b_opt = res_b.x.reshape(n_train, N_SP)
        print(f'  Step2 b: loss={res_b.fun:.5f}  iters={res_b.nit}'
              f'  ({time.time()-t_fold:.1f}s)', flush=True)

        # Train RMSE (single batched call)
        pw2_tr = batch_np(A_opt, b_opt, phi_train_np[:, 0, :])
        pw3_tr = batch_np(A_opt, b_opt, pw2_tr)
        m2_tr  = mask_train_np[:, 1]; m3_tr = mask_train_np[:, 2]
        sq_tr  = np.sum(m2_tr[:, None] * (pw2_tr - phi_train_np[:, 1, :])**2)
        sq_tr += np.sum(m3_tr[:, None] * (pw3_tr - phi_train_np[:, 2, :])**2)
        cnt_tr = np.sum(m2_tr + m3_tr) * N_SP
        train_rmse = float(np.sqrt(sq_tr / cnt_tr)) if cnt_tr > 0 else float('nan')
        print(f'  Train RMSE: {train_rmse:.5f}', flush=True)

        # ── Step 3: Refit b_test from W1→W2 ────────────────────────────
        res_bt = minimize(
            loss_b_test,
            x0=np.full(N_SP, 0.1),
            args=(A_opt, phi_test_np, mask_test_np, single_np, lam_b),
            method='L-BFGS-B',
            options=dict(maxiter=500, maxfun=2000, ftol=1e-10, gtol=1e-7),
        )
        b_test = res_bt.x
        print(f'  Step3 b_test: loss={res_bt.fun:.5f}  iters={res_bt.nit}'
              f'  ({time.time()-t_fold:.1f}s)', flush=True)

        # ── Predict W3 ────────────────────────────────────────────────
        pw2_pred = single_np(A_opt, b_test, phi_test_np[0])
        pw3_pred = single_np(A_opt, b_test, pw2_pred)
        phi_W3_obs = phi_test_np[2]
        m3 = mask_test_np[2]

        loo_rmse = (float(np.sqrt(np.mean((pw3_pred - phi_W3_obs)**2)))
                    if m3 > 0 else float('nan'))
        print(f'  LOO RMSE (W3): {loo_rmse:.5f}  ({time.time()-t_fold:.1f}s)', flush=True)
        print(f'  W3 pred: {pw3_pred.round(3)}', flush=True)
        print(f'  W3 obs:  {phi_W3_obs.round(3)}', flush=True)

        results.append({'patient': test_pat, 'rmse': loo_rmse, 'train_rmse': train_rmse})

    valid     = [r['rmse'] for r in results if not np.isnan(r['rmse'])]
    mean_rmse = float(np.mean(valid)) if valid else float('nan')
    print(f'\n{"="*60}', flush=True)
    print(f'NSP full LOO mean RMSE: {mean_rmse:.5f}', flush=True)
    print(f'gLV Replicator baseline:  0.05885', flush=True)
    print(f'NSP approx (A fixed):     0.08551', flush=True)
    for r in results:
        print(f"  {r['patient']}: {r['rmse']:.5f}  (train={r['train_rmse']:.5f})", flush=True)

    return results, mean_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi-npy',   default=str(_here / 'results' / 'dieckow_otu' / 'phi_guild.npy'))
    ap.add_argument('--out-json',  default=str(_here / 'results' / 'dieckow_cr' / 'loo_nsp_full_gpu.json'))
    ap.add_argument('--n-steps',   type=int,   default=100)
    ap.add_argument('--dt',        type=float, default=0.01,
                    help='ODE time step (dt=0.01, n_steps=100 → total 1.0/week, same as gLV)')
    ap.add_argument('--maxiter',   type=int,   default=200,  help='L-BFGS-B max iterations')
    ap.add_argument('--maxfun',    type=int,   default=2000, help='L-BFGS-B max function evals')
    ap.add_argument('--lambda-a',  type=float, default=1e-4)
    ap.add_argument('--lambda-b',  type=float, default=1e-3)
    args = ap.parse_args()

    print(f'JAX devices: {jax.devices()}', flush=True)
    print(f'n_steps={args.n_steps}  dt={args.dt}  total_time_per_week={args.n_steps*args.dt:.3f}', flush=True)
    print(f'maxiter={args.maxiter}  maxfun={args.maxfun}', flush=True)
    print(f'λA={args.lambda_a}  λb={args.lambda_b}', flush=True)

    phi_all = np.load(args.phi_npy)
    assert phi_all.ndim == 3 and phi_all.shape[1] == 3
    n_p_raw, _, n_sp = phi_all.shape
    assert n_sp == N_SP

    keep     = phi_all[:, 0, :].sum(axis=1) > 1e-12
    phi_all  = phi_all[keep]
    patients = ([p for k, p in zip(keep.tolist(), PAT_ALL) if k]
                if n_p_raw == 10 else [str(i) for i in range(phi_all.shape[0])])
    print(f'Patients ({len(patients)}): {patients}', flush=True)

    results, mean_rmse = run_loo(phi_all, patients, args)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        'loo_rmse_mean': mean_rmse,
        'per_patient':   results,
        'model': (f'NSP Hamilton full LOO-CV '
                  f'(forward-only JAX + scipy L-BFGS-B, n_steps={args.n_steps}, dt={args.dt})'),
        'n_steps':   args.n_steps,
        'dt':        args.dt,
        'total_time_per_week': args.n_steps * args.dt,
        'maxfun':    args.maxfun,
        'guilds':    GUILD_ORDER,
    }, open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
