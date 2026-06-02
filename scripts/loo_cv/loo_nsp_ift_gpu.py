#!/usr/bin/env python3
"""
loo_nsp_ift_gpu.py — LOO-CV for NSP Hamilton ODE with IFT gradient.

Uses IFT custom_vjp (Implicit Function Theorem) on Newton step so that
value_and_grad compiles in ~30-60s instead of 600s+.

Gradient per step = 1 JAX call (vs 112 scipy FD calls in loo_nsp_full_gpu.py).
Expected: ~5-10× faster optimization per fold.

Strategy per fold (identical to loo_nsp_full_gpu.py):
  1. Optimize A_upper (n_A=55 params) with b fixed — JAX grad via IFT
  2. Optimize b_train (n_train×N_SP) with A fixed — JAX grad via IFT
  3. Refit b_test (N_SP params) from W1→W2 — JAX grad via IFT
  4. Predict W3

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 loo_nsp_ift_gpu.py --fold-start 0 --fold-end 4
  CUDA_VISIBLE_DEVICES=1 python3 loo_nsp_ift_gpu.py --fold-start 4 --fold-end 7
  CUDA_VISIBLE_DEVICES=2 python3 loo_nsp_ift_gpu.py --fold-start 7 --fold-end 10
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import argparse, json, sys, time, os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('TMPDIR', str(Path.home() / 'tmp'))

import jax
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad

jax.config.update('jax_enable_x64', True)

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))

from guild_replicator_dieckow import GUILD_ORDER
from hamilton_ode_jax_nsp_ift import make_ift_simulate

N_SP = len(GUILD_ORDER)
N_A  = N_SP * (N_SP + 1) // 2   # 55
PAT_ALL = list('ABCDEFGHKL')

_DIAG_IDX = [j * (j + 1) // 2 + j for j in range(N_SP)]


def default_A_upper():
    A = -0.1 * np.eye(N_SP)
    return np.array([A[i, j] for j in range(N_SP) for i in range(j + 1)])


def full_to_A_upper(A):
    return np.array([A[i, j] for j in range(N_SP) for i in range(j + 1)])


def _clip_diag_np(A_upper):
    A_upper = np.array(A_upper)
    for idx in _DIAG_IDX:
        A_upper[idx] = min(A_upper[idx], 0.0)
    return A_upper


# ── Build JAX loss functions with IFT gradient ────────────────────────────────

def make_loss_fns(simulate_single, lam_A, lam_b):
    """Build JIT-compiled (value, grad) functions for A, b_train, b_test."""

    simulate_batch = vmap(simulate_single, in_axes=(None, 0, 0))

    # ── Loss A: optimize A_upper, b_all fixed ────────────────────────────────
    def _loss_A(A_upper, b_all, phi_train, mask_train):
        # phi_train: (n_train, 3, N_SP), mask_train: (n_train, 3)
        pw2 = simulate_batch(A_upper, b_all, phi_train[:, 0, :])
        pw3 = simulate_batch(A_upper, b_all, pw2)
        sq  = jnp.sum(mask_train[:, 1, None] * (pw2 - phi_train[:, 1, :]) ** 2)
        sq += jnp.sum(mask_train[:, 2, None] * (pw3 - phi_train[:, 2, :]) ** 2)
        cnt = jnp.sum(mask_train[:, 1] + mask_train[:, 2]) * N_SP
        rmse = jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0))
        return rmse + lam_A * jnp.sum(A_upper ** 2)

    vg_A = jit(value_and_grad(_loss_A, argnums=0))

    # ── Loss b: optimize b_all (passed flat by scipy), A fixed ───────────────
    def _loss_b(b_flat, A_upper, phi_train, mask_train):
        b_all = b_flat.reshape(phi_train.shape[0], N_SP)
        pw2 = simulate_batch(A_upper, b_all, phi_train[:, 0, :])
        pw3 = simulate_batch(A_upper, b_all, pw2)
        sq  = jnp.sum(mask_train[:, 1, None] * (pw2 - phi_train[:, 1, :]) ** 2)
        sq += jnp.sum(mask_train[:, 2, None] * (pw3 - phi_train[:, 2, :]) ** 2)
        cnt = jnp.sum(mask_train[:, 1] + mask_train[:, 2]) * N_SP
        rmse = jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0))
        return rmse + lam_b * jnp.sum(b_all ** 2)

    vg_b = jit(value_and_grad(_loss_b, argnums=0))

    # ── Loss b_test: optimize b_p for one patient (W1→W2) ────────────────────
    def _loss_bt(b_p, A_upper, phi_test, mask_test):
        pw2 = simulate_single(A_upper, b_p, phi_test[0])
        sq  = mask_test[1] * jnp.sum((pw2 - phi_test[1]) ** 2)
        cnt = mask_test[1] * N_SP
        rmse = jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0))
        return rmse + lam_b * jnp.sum(b_p ** 2)

    vg_bt = jit(value_and_grad(_loss_bt, argnums=0))

    # ── Forward-only helpers (no grad needed) ────────────────────────────────
    fwd_batch  = jit(simulate_batch)
    fwd_single = jit(simulate_single)

    return vg_A, vg_b, vg_bt, fwd_batch, fwd_single


# ── scipy wrappers ────────────────────────────────────────────────────────────

def _jax_to_scipy(vg_fn, *static_args, verbose=False):
    """Return a fun(x_np) → (val, grad_np) callable for scipy L-BFGS-B."""
    call_count = [0]
    def fn(x_np):
        val, grad = vg_fn(jnp.array(x_np, dtype=jnp.float64), *static_args)
        g = np.array(grad, dtype=np.float64)
        if verbose and call_count[0] < 3:
            print(f'    [eval {call_count[0]}] loss={float(val):.6f}  |grad|={np.linalg.norm(g):.3e}  '
                  f'|grad_max|={np.abs(g).max():.3e}', flush=True)
        call_count[0] += 1
        return float(val), g
    return fn


# ── LOO-CV ────────────────────────────────────────────────────────────────────

def run_loo(phi_all, patients, args):
    n_p  = len(patients)
    mask = (phi_all.sum(axis=2) > 1e-12).astype(np.float64)

    # Build IFT simulate function
    simulate_single = make_ift_simulate(
        n_sp=N_SP, n_steps=args.n_steps, dt=args.dt,
        c_const=25.0, alpha_const=100.0,
    )

    lam_A = args.lambda_a
    lam_b = args.lambda_b
    vg_A, vg_b, vg_bt, fwd_batch, fwd_single = make_loss_fns(
        simulate_single, lam_A, lam_b
    )

    # ── Compile (dry-run with n_train=n_p-1) ─────────────────────────────────
    n_train0 = n_p - 1
    _phi0 = jnp.ones((n_train0, 3, N_SP), dtype=jnp.float64) / N_SP
    _mask0 = jnp.ones((n_train0, 3), dtype=jnp.float64)
    _b0   = jnp.zeros((n_train0, N_SP), dtype=jnp.float64)
    _A0   = jnp.array(default_A_upper(), dtype=jnp.float64)
    _phi1 = jnp.ones((3, N_SP), dtype=jnp.float64) / N_SP
    _mask1 = jnp.ones(3, dtype=jnp.float64)

    print('\nJIT-compiling IFT value_and_grad (expect 30-120s)...', flush=True)
    t0 = time.time()
    _ = vg_A(_A0, _b0, _phi0, _mask0)
    print(f'  vg_A compiled: {time.time()-t0:.1f}s', flush=True)
    _ = vg_b(_b0, _A0, _phi0, _mask0)
    print(f'  vg_b compiled: {time.time()-t0:.1f}s', flush=True)
    _ = vg_bt(jnp.zeros(N_SP, dtype=jnp.float64), _A0, _phi1, _mask1)
    print(f'  vg_bt compiled: {time.time()-t0:.1f}s', flush=True)
    _ = fwd_batch(_A0, _b0, _phi0[:, 0, :])
    print(f'  fwd_batch compiled: {time.time()-t0:.1f}s', flush=True)

    # Warm start
    warm_paths = [
        _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked_v2.json',
        _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json',
        _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton.json',
    ]
    warm_A = None; warm_b_map = {}
    for wp in warm_paths:
        if not wp.exists(): continue
        d   = json.load(open(wp))
        A0  = np.array(d['A'])
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

    a_bounds = [(-np.inf, 0.0) if k in _DIAG_IDX else (-np.inf, np.inf)
                for k in range(N_A)]

    fold_range = range(args.fold_start, args.fold_end)
    results = []

    for fold in fold_range:
        test_pat = patients[fold]
        print(f'\n{"="*60}', flush=True)
        print(f'Fold {fold+1}/{n_p}  leave-out: {test_pat}', flush=True)
        t_fold = time.time()

        train_idx = [i for i, p in enumerate(patients) if p != test_pat]
        test_idx  = patients.index(test_pat)
        n_train   = len(train_idx)

        phi_train_j  = jnp.array(phi_all[np.array(train_idx)], dtype=jnp.float64)
        mask_train_j = jnp.array(mask[np.array(train_idx)],    dtype=jnp.float64)
        phi_test_j   = jnp.array(phi_all[test_idx],            dtype=jnp.float64)
        mask_test_j  = jnp.array(mask[test_idx],               dtype=jnp.float64)

        b_warm = jnp.array(
            [warm_b_map.get(patients[i], np.full(N_SP, 0.1)) for i in train_idx],
            dtype=jnp.float64,
        )
        A_warm = jnp.array(warm_A, dtype=jnp.float64)

        # ── Step 1: Optimise A_upper (IFT grad) ──────────────────────────
        verbose1 = (fold == args.fold_start)  # print grad info only for first fold
        A_fun = _jax_to_scipy(vg_A, b_warm, phi_train_j, mask_train_j, verbose=verbose1)
        res_A = minimize(
            A_fun,
            x0=np.array(A_warm),
            method='L-BFGS-B',
            jac=True,
            bounds=a_bounds,
            options=dict(maxiter=args.maxiter, maxfun=args.maxfun,
                         ftol=1e-12, gtol=1e-9),
        )
        A_opt = jnp.array(_clip_diag_np(res_A.x), dtype=jnp.float64)
        print(f'  Step1 A: loss={res_A.fun:.5f}  iters={res_A.nit}'
              f'  ({time.time()-t_fold:.1f}s)', flush=True)

        # ── Step 2: Optimise b_train (IFT grad) ──────────────────────────
        # Start from zeros: alpha=100 makes b gradients O(100) at b_warm,
        # which breaks L-BFGS-B line search. At b=0 the gradient is smaller.
        b_fun = _jax_to_scipy(vg_b, A_opt, phi_train_j, mask_train_j, verbose=verbose1)
        res_b = minimize(
            b_fun,
            x0=np.zeros(n_train * N_SP),
            method='L-BFGS-B',
            jac=True,
            options=dict(maxiter=args.maxiter, maxfun=args.maxfun,
                         ftol=1e-12, gtol=1e-9),
        )
        b_opt = jnp.array(res_b.x.reshape(n_train, N_SP), dtype=jnp.float64)
        print(f'  Step2 b: loss={res_b.fun:.5f}  iters={res_b.nit}'
              f'  ({time.time()-t_fold:.1f}s)', flush=True)

        # Train RMSE
        pw2_tr = np.array(fwd_batch(A_opt, b_opt, phi_train_j[:, 0, :]))
        pw3_tr = np.array(fwd_batch(A_opt, b_opt, jnp.array(pw2_tr)))
        phi_train_np  = np.array(phi_train_j)
        mask_train_np = np.array(mask_train_j)
        m2_tr = mask_train_np[:, 1]; m3_tr = mask_train_np[:, 2]
        sq_tr  = np.sum(m2_tr[:, None] * (pw2_tr - phi_train_np[:, 1, :]) ** 2)
        sq_tr += np.sum(m3_tr[:, None] * (pw3_tr - phi_train_np[:, 2, :]) ** 2)
        cnt_tr = np.sum(m2_tr + m3_tr) * N_SP
        train_rmse = float(np.sqrt(sq_tr / cnt_tr)) if cnt_tr > 0 else float('nan')
        print(f'  Train RMSE: {train_rmse:.5f}', flush=True)

        # ── Step 3: Refit b_test (IFT grad) ──────────────────────────────
        bt_fun = _jax_to_scipy(vg_bt, A_opt, phi_test_j, mask_test_j, verbose=verbose1)
        res_bt = minimize(
            bt_fun,
            x0=np.full(N_SP, 0.1),
            method='L-BFGS-B',
            jac=True,
            options=dict(maxiter=500, maxfun=2000, ftol=1e-12, gtol=1e-9),
        )
        b_test = jnp.array(res_bt.x, dtype=jnp.float64)
        print(f'  Step3 b_test: loss={res_bt.fun:.5f}  iters={res_bt.nit}'
              f'  ({time.time()-t_fold:.1f}s)', flush=True)

        # ── Predict W3 ────────────────────────────────────────────────────
        pw2_pred = np.array(fwd_single(A_opt, b_test, phi_test_j[0]))
        pw3_pred = np.array(fwd_single(A_opt, b_test, jnp.array(pw2_pred)))
        phi_W3_obs = np.array(phi_test_j[2])
        m3 = float(mask_test_j[2])

        loo_rmse = (float(np.sqrt(np.mean((pw3_pred - phi_W3_obs) ** 2)))
                    if m3 > 0 else float('nan'))
        print(f'  LOO RMSE (W3): {loo_rmse:.5f}  ({time.time()-t_fold:.1f}s)', flush=True)
        print(f'  W3 pred: {pw3_pred.round(3)}', flush=True)
        print(f'  W3 obs:  {phi_W3_obs.round(3)}', flush=True)

        results.append({'patient': test_pat, 'fold': fold,
                        'rmse': loo_rmse, 'train_rmse': train_rmse})

    valid     = [r['rmse'] for r in results if not np.isnan(r['rmse'])]
    mean_rmse = float(np.mean(valid)) if valid else float('nan')
    print(f'\n{"="*60}', flush=True)
    print(f'NSP IFT LOO mean RMSE (folds {args.fold_start}-{args.fold_end-1}): {mean_rmse:.5f}', flush=True)
    print(f'gLV Replicator baseline:  0.05885', flush=True)
    print(f'NSP approx (A fixed):     0.08551', flush=True)
    for r in results:
        print(f"  {r['patient']}: {r['rmse']:.5f}  (train={r['train_rmse']:.5f})", flush=True)

    return results, mean_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi-npy',    default=str(_here / 'results' / 'dieckow_otu' / 'phi_guild.npy'))
    ap.add_argument('--out-json',   default=str(_here / 'results' / 'dieckow_cr' / 'loo_nsp_ift_gpu.json'))
    ap.add_argument('--n-steps',    type=int,   default=100)
    ap.add_argument('--dt',         type=float, default=0.01,
                    help='ODE time step (dt=0.01, n_steps=100 → total 1.0/week)')
    ap.add_argument('--maxiter',    type=int,   default=200)
    ap.add_argument('--maxfun',     type=int,   default=2000)
    ap.add_argument('--lambda-a',   type=float, default=1e-4)
    ap.add_argument('--lambda-b',   type=float, default=1e-3)
    ap.add_argument('--fold-start', type=int,   default=0,
                    help='First fold index to run (inclusive)')
    ap.add_argument('--fold-end',   type=int,   default=None,
                    help='Last fold index (exclusive); defaults to n_patients')
    args = ap.parse_args()

    print(f'JAX devices: {jax.devices()}', flush=True)
    print(f'n_steps={args.n_steps}  dt={args.dt}  total_time/week={args.n_steps*args.dt:.3f}', flush=True)
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
    n_p = len(patients)

    if args.fold_end is None:
        args.fold_end = n_p
    args.fold_end = min(args.fold_end, n_p)

    print(f'Patients ({n_p}): {patients}', flush=True)
    print(f'Running folds {args.fold_start} to {args.fold_end - 1}', flush=True)

    results, mean_rmse = run_loo(phi_all, patients, args)

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Load existing results to merge if partial run
    existing = []
    if out.exists():
        try:
            existing = json.load(open(out)).get('per_patient', [])
        except Exception:
            pass
    # Replace entries for folds we just ran
    ran_folds = {r['fold'] for r in results}
    merged = [e for e in existing if e.get('fold') not in ran_folds] + results
    merged.sort(key=lambda r: r.get('fold', 0))
    all_valid  = [r['rmse'] for r in merged if not np.isnan(r['rmse'])]
    all_mean   = float(np.mean(all_valid)) if all_valid else float('nan')

    json.dump({
        'loo_rmse_mean': all_mean,
        'per_patient':   merged,
        'model': (f'NSP Hamilton IFT LOO-CV '
                  f'(JAX value_and_grad + IFT custom_vjp, n_steps={args.n_steps}, dt={args.dt})'),
        'n_steps':             args.n_steps,
        'dt':                  args.dt,
        'total_time_per_week': args.n_steps * args.dt,
        'maxfun':              args.maxfun,
        'guilds':              GUILD_ORDER,
        'folds_completed':     sorted(r['fold'] for r in merged),
    }, open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)
    print(f'All-fold mean RMSE (from JSON, {len(merged)} folds): {all_mean:.5f}', flush=True)


if __name__ == '__main__':
    main()
