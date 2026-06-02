#!/usr/bin/env python3
"""
loo_nsp_ift_v4_gpu.py — NSP Hamilton ODE LOO-CV v4: joint A+b optimization

v3 showed that alternating A/b rounds oscillate (non-convex landscape) without
converging. v4 solves this by optimizing A_upper and b_train jointly in a single
L-BFGS-B call over the concatenated parameter vector [A_upper (55), b_flat (90)].

Changes from v2:
  - Joint optimization of A+b (no coordinate descent)
  - maxfun=5000
  - Warm-start A from fit_guild_hamilton.json (same as v2), b from zeros

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 loo_nsp_ift_v4_gpu.py --fold-start 0 --fold-end 5
  CUDA_VISIBLE_DEVICES=1 python3 loo_nsp_ift_v4_gpu.py --fold-start 5 --fold-end 10
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


def make_loss_fns(simulate_single, lam_A, lam_b, n_train):
    simulate_batch = vmap(simulate_single, in_axes=(None, 0, 0))

    # Joint loss: params = [A_upper (N_A), b_flat (n_train * N_SP)]
    def _loss_joint(params, phi_train, mask_train):
        A_upper = params[:N_A]
        b_all   = params[N_A:].reshape(n_train, N_SP)
        pw2 = simulate_batch(A_upper, b_all, phi_train[:, 0, :])
        pw3 = simulate_batch(A_upper, b_all, pw2)
        sq  = jnp.sum(mask_train[:, 1, None] * (pw2 - phi_train[:, 1, :]) ** 2)
        sq += jnp.sum(mask_train[:, 2, None] * (pw3 - phi_train[:, 2, :]) ** 2)
        cnt = jnp.sum(mask_train[:, 1] + mask_train[:, 2]) * N_SP
        rmse = jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0))
        return rmse + lam_A * jnp.sum(A_upper ** 2) + lam_b * jnp.sum(b_all ** 2)

    vg_joint = jit(value_and_grad(_loss_joint, argnums=0))

    # b_test loss: optimize b for one test patient (W1→W2)
    def _loss_bt(b_p, A_upper, phi_test, mask_test):
        pw2 = simulate_single(A_upper, b_p, phi_test[0])
        sq  = mask_test[1] * jnp.sum((pw2 - phi_test[1]) ** 2)
        cnt = mask_test[1] * N_SP
        rmse = jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0))
        return rmse + lam_b * jnp.sum(b_p ** 2)

    vg_bt = jit(value_and_grad(_loss_bt, argnums=0))

    fwd_batch  = jit(simulate_batch)
    fwd_single = jit(simulate_single)

    return vg_joint, vg_bt, fwd_batch, fwd_single


def _jax_to_scipy(vg_fn, *static_args, verbose=False):
    call_count = [0]
    def fn(x_np):
        val, grad = vg_fn(jnp.array(x_np, dtype=jnp.float64), *static_args)
        g = np.array(grad, dtype=np.float64)
        if verbose and call_count[0] < 3:
            print(f'    [eval {call_count[0]}] loss={float(val):.6f}  |grad|={np.linalg.norm(g):.3e}  '
                  f'grad_A={np.linalg.norm(g[:N_A]):.2e}  grad_b={np.linalg.norm(g[N_A:]):.2e}',
                  flush=True)
        call_count[0] += 1
        return float(val), g
    return fn


def run_loo(phi_all, patients, args):
    n_p  = len(patients)
    mask = (phi_all.sum(axis=2) > 1e-12).astype(np.float64)
    n_train_std = n_p - 1  # 9 for 10-patient dataset

    simulate_single = make_ift_simulate(
        n_sp=N_SP, n_steps=args.n_steps, dt=args.dt,
        c_const=25.0, alpha_const=100.0,
    )

    lam_A = args.lambda_a
    lam_b = args.lambda_b
    vg_joint, vg_bt, fwd_batch, fwd_single = make_loss_fns(
        simulate_single, lam_A, lam_b, n_train_std
    )

    # Compile
    _phi0  = jnp.ones((n_train_std, 3, N_SP), dtype=jnp.float64) / N_SP
    _mask0 = jnp.ones((n_train_std, 3), dtype=jnp.float64)
    _p0    = jnp.zeros(N_A + n_train_std * N_SP, dtype=jnp.float64)
    _A0    = jnp.array(default_A_upper(), dtype=jnp.float64)
    _phi1  = jnp.ones((3, N_SP), dtype=jnp.float64) / N_SP
    _mask1 = jnp.ones(3, dtype=jnp.float64)

    print('\nJIT-compiling IFT value_and_grad (expect 60-180s)...', flush=True)
    t0 = time.time()
    _ = vg_joint(_p0, _phi0, _mask0)
    print(f'  vg_joint compiled: {time.time()-t0:.1f}s', flush=True)
    _ = vg_bt(jnp.zeros(N_SP, dtype=jnp.float64), _A0, _phi1, _mask1)
    print(f'  vg_bt compiled: {time.time()-t0:.1f}s', flush=True)
    _ = fwd_batch(_A0, jnp.zeros((n_train_std, N_SP), dtype=jnp.float64), _phi0[:, 0, :])
    print(f'  fwd_batch compiled: {time.time()-t0:.1f}s', flush=True)

    # Load warm A — prefer fit_guild_hamilton.json (same as v2 baseline)
    warm_paths = [
        _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton.json',
        _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked_v2_10g.json',
        _here / 'results' / 'dieckow_cr' / 'fit_guild_hamilton_masked.json',
    ]
    warm_A = None
    for wp in warm_paths:
        if not wp.exists(): continue
        d  = json.load(open(wp))
        A0 = np.array(d['A'])
        if A0.shape[0] != N_SP: continue
        A_sym  = (A0 + A0.T) / 2.0
        warm_A = full_to_A_upper(A_sym)
        print(f'Warm-start A from {wp.name}', flush=True)
        break
    if warm_A is None:
        warm_A = default_A_upper()
        print('Warm-start A: default (-0.1 diagonal)', flush=True)

    # Joint bounds: diag of A_upper <= 0, everything else unbounded
    joint_bounds = (
        [(-np.inf, 0.0) if k in _DIAG_IDX else (-np.inf, np.inf) for k in range(N_A)]
        + [(-np.inf, np.inf)] * (n_train_std * N_SP)
    )

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
        assert n_train == n_train_std, f'Expected {n_train_std} train patients, got {n_train}'

        phi_train_j  = jnp.array(phi_all[np.array(train_idx)], dtype=jnp.float64)
        mask_train_j = jnp.array(mask[np.array(train_idx)],    dtype=jnp.float64)
        phi_test_j   = jnp.array(phi_all[test_idx],            dtype=jnp.float64)
        mask_test_j  = jnp.array(mask[test_idx],               dtype=jnp.float64)

        # Init: [warm_A, zeros(b)] — zeros for b avoids alpha=100 gradient explosion
        x0 = np.concatenate([warm_A, np.zeros(n_train * N_SP)])

        verbose_fold = (fold == args.fold_start)
        joint_fun = _jax_to_scipy(vg_joint, phi_train_j, mask_train_j, verbose=verbose_fold)

        res = minimize(
            joint_fun,
            x0=x0,
            method='L-BFGS-B',
            jac=True,
            bounds=joint_bounds,
            options=dict(maxiter=args.maxiter, maxfun=args.maxfun,
                         ftol=1e-12, gtol=1e-9),
        )

        A_opt = jnp.array(_clip_diag_np(res.x[:N_A]), dtype=jnp.float64)
        b_opt = jnp.array(res.x[N_A:].reshape(n_train, N_SP), dtype=jnp.float64)
        print(f'  Joint opt: loss={res.fun:.5f}  iters={res.nit}'
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

        # Step 3: Refit b_test
        bt_fun = _jax_to_scipy(vg_bt, A_opt, phi_test_j, mask_test_j, verbose=verbose_fold)
        res_bt = minimize(
            bt_fun,
            x0=np.full(N_SP, 0.1),
            method='L-BFGS-B',
            jac=True,
            options=dict(maxiter=500, maxfun=5000, ftol=1e-12, gtol=1e-9),
        )
        b_test = jnp.array(res_bt.x, dtype=jnp.float64)
        print(f'  Step3 b_test: loss={res_bt.fun:.5f}  iters={res_bt.nit}'
              f'  ({time.time()-t_fold:.1f}s)', flush=True)

        # Predict W3
        pw2_pred = np.array(fwd_single(A_opt, b_test, phi_test_j[0]))
        pw3_pred = np.array(fwd_single(A_opt, b_test, jnp.array(pw2_pred)))
        phi_W3_obs = np.array(phi_test_j[2])
        m3 = float(mask_test_j[2])

        loo_rmse = (float(np.sqrt(np.mean((pw3_pred - phi_W3_obs) ** 2)))
                    if m3 > 0 else float('nan'))
        loo_bc   = (0.5 * float(np.sum(np.abs(pw3_pred - phi_W3_obs)))
                    if m3 > 0 else float('nan'))
        print(f'  LOO RMSE (W3): {loo_rmse:.5f}  BC: {loo_bc:.5f}  ({time.time()-t_fold:.1f}s)', flush=True)
        print(f'  W3 pred: {pw3_pred.round(3)}', flush=True)
        print(f'  W3 obs:  {phi_W3_obs.round(3)}', flush=True)

        results.append({'patient': test_pat, 'fold': fold,
                        'rmse': loo_rmse, 'bc': loo_bc, 'train_rmse': train_rmse})

    valid     = [r['rmse'] for r in results if not np.isnan(r['rmse'])]
    mean_rmse = float(np.mean(valid)) if valid else float('nan')
    print(f'\n{"="*60}', flush=True)
    valid_bc  = [r['bc']   for r in results if not np.isnan(r.get('bc', float('nan')))]
    mean_bc   = float(np.mean(valid_bc)) if valid_bc else float('nan')
    print(f'NSP IFT v4 LOO mean RMSE: {mean_rmse:.5f}  BC: {mean_bc:.5f}', flush=True)
    print(f'gLV (2-step):  RMSE=0.04502  BC=0.14409', flush=True)
    print(f'NSP approx:    RMSE=0.08551', flush=True)
    print(f'NSP IFT v2:    RMSE=0.09055  BC=0.28905', flush=True)
    for r in results:
        print(f"  {r['patient']}: RMSE={r['rmse']:.5f}  BC={r.get('bc', float('nan')):.5f}"
              f"  (train={r['train_rmse']:.5f})", flush=True)

    return results, mean_rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi-npy',    default=str(_here / 'results' / 'dieckow_otu' / 'phi_guild.npy'))
    ap.add_argument('--out-json',   default=None)
    ap.add_argument('--n-steps',    type=int,   default=100)
    ap.add_argument('--dt',         type=float, default=0.01)
    ap.add_argument('--maxiter',    type=int,   default=1000)
    ap.add_argument('--maxfun',     type=int,   default=5000)
    ap.add_argument('--lambda-a',   type=float, default=1e-4)
    ap.add_argument('--lambda-b',   type=float, default=1e-3)
    ap.add_argument('--fold-start', type=int,   default=0)
    ap.add_argument('--fold-end',   type=int,   default=None)
    args = ap.parse_args()

    if args.out_json is None:
        tag = f'folds{args.fold_start}-{(args.fold_end or 10)-1}'
        args.out_json = str(_here / 'results' / 'dieckow_cr' /
                            f'loo_nsp_ift_v4_{tag}.json')

    print(f'JAX devices: {jax.devices()}', flush=True)
    print(f'n_steps={args.n_steps}  dt={args.dt}  joint_opt=True', flush=True)
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
    existing = []
    if out.exists():
        try:
            existing = json.load(open(out)).get('per_patient', [])
        except Exception:
            pass
    ran_folds = {r['fold'] for r in results}
    merged = [e for e in existing if e.get('fold') not in ran_folds] + results
    merged.sort(key=lambda r: r.get('fold', 0))
    all_valid  = [r['rmse'] for r in merged if not np.isnan(r['rmse'])]
    all_mean   = float(np.mean(all_valid)) if all_valid else float('nan')

    all_bc_valid = [r['bc'] for r in merged if not np.isnan(r.get('bc', float('nan')))]
    all_bc_mean  = float(np.mean(all_bc_valid)) if all_bc_valid else float('nan')

    json.dump({
        'loo_rmse_mean':   all_mean,
        'loo_bc_mean':     all_bc_mean,
        'per_patient':     merged,
        'model':           (f'NSP Hamilton IFT v4 LOO-CV '
                            f'(joint A+b, n_steps={args.n_steps}, dt={args.dt}, '
                            f'maxfun={args.maxfun})'),
        'n_steps':         args.n_steps,
        'dt':              args.dt,
        'maxfun':          args.maxfun,
        'lambda_a':        args.lambda_a,
        'lambda_b':        args.lambda_b,
        'guilds':          GUILD_ORDER,
        'folds_completed': sorted(r['fold'] for r in merged),
    }, open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)
    print(f'All-fold mean RMSE ({len(merged)} folds): {all_mean:.5f}', flush=True)


if __name__ == '__main__':
    main()
