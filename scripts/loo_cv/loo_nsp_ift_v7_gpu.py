#!/usr/bin/env python3
"""
loo_nsp_ift_v7_gpu.py — NSP Hamilton IFT LOO-CV with mixed CLR+MSE loss

Changes from v6:
  - Mixed loss: alpha_clr * CLR_RMSE + (1-alpha_clr) * MSE_RMSE
  - maxfun=5000 (was 2000 in v6)
  - --alpha-clr controls the blend (0=pure MSE=v2, 1=pure CLR=v6)

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 loo_nsp_ift_v7_gpu.py --alpha-clr 0.3 --fold-start 0 --fold-end 5
  CUDA_VISIBLE_DEVICES=1 python3 loo_nsp_ift_v7_gpu.py --alpha-clr 0.3 --fold-start 5 --fold-end 10
  CUDA_VISIBLE_DEVICES=2 python3 loo_nsp_ift_v7_gpu.py --alpha-clr 0.7 --fold-start 0 --fold-end 5
  CUDA_VISIBLE_DEVICES=3 python3 loo_nsp_ift_v7_gpu.py --alpha-clr 0.7 --fold-start 5 --fold-end 10
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

N_SP      = len(GUILD_ORDER)
N_A       = N_SP * (N_SP + 1) // 2
PAT_ALL   = list('ABCDEFGHKL')
_DIAG_IDX = [j * (j + 1) // 2 + j for j in range(N_SP)]
CLR_EPS   = 1e-6


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


# ── Mixed loss helpers ─────────────────────────────────────────────────────────

def _clr(phi):
    lp = jnp.log(phi + CLR_EPS)
    return lp - lp.mean(axis=-1, keepdims=True)

def _mixed_sq(pred, obs, mask, alpha_clr):
    """alpha_clr * CLR_sq + (1-alpha_clr) * MSE_sq, summed over patients×guilds."""
    mse_sq = jnp.sum(mask[:, None] * (pred - obs) ** 2)
    clr_sq = jnp.sum(mask[:, None] * (_clr(pred) - _clr(obs)) ** 2)
    return alpha_clr * clr_sq + (1.0 - alpha_clr) * mse_sq


def make_loss_fns(simulate_single, lam_A, lam_A_l1, lam_b, alpha_clr):
    simulate_batch = vmap(simulate_single, in_axes=(None, 0, 0))

    def _loss_A(A_upper, b_all, phi_train, mask_train):
        pw2 = simulate_batch(A_upper, b_all, phi_train[:, 0, :])
        pw3 = simulate_batch(A_upper, b_all, pw2)
        sq  = _mixed_sq(pw2, phi_train[:, 1, :], mask_train[:, 1], alpha_clr)
        sq += _mixed_sq(pw3, phi_train[:, 2, :], mask_train[:, 2], alpha_clr)
        cnt = jnp.sum(mask_train[:, 1] + mask_train[:, 2]) * N_SP
        loss = jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0))
        return loss + lam_A * jnp.sum(A_upper**2) + lam_A_l1 * jnp.sum(jnp.abs(A_upper))

    vg_A = jit(value_and_grad(_loss_A, argnums=0))

    def _loss_b(b_flat, A_upper, phi_train, mask_train):
        b_all = b_flat.reshape(phi_train.shape[0], N_SP)
        pw2 = simulate_batch(A_upper, b_all, phi_train[:, 0, :])
        pw3 = simulate_batch(A_upper, b_all, pw2)
        sq  = _mixed_sq(pw2, phi_train[:, 1, :], mask_train[:, 1], alpha_clr)
        sq += _mixed_sq(pw3, phi_train[:, 2, :], mask_train[:, 2], alpha_clr)
        cnt = jnp.sum(mask_train[:, 1] + mask_train[:, 2]) * N_SP
        return jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0)) + lam_b * jnp.sum(b_flat**2)

    vg_b = jit(value_and_grad(_loss_b, argnums=0))

    def _loss_bt(b_p, A_upper, phi_test, mask_test):
        pw2 = simulate_single(A_upper, b_p, phi_test[0])
        sq  = _mixed_sq(pw2[None], phi_test[1][None], mask_test[1][None], alpha_clr)
        cnt = mask_test[1] * N_SP
        return jnp.sqrt(jnp.where(cnt > 0, sq / cnt, 0.0)) + lam_b * jnp.sum(b_p**2)

    vg_bt      = jit(value_and_grad(_loss_bt, argnums=0))
    fwd_batch  = jit(simulate_batch)
    fwd_single = jit(simulate_single)
    return vg_A, vg_b, vg_bt, fwd_batch, fwd_single


def _jax_to_scipy(vg_fn, *static_args, verbose=False):
    cc = [0]
    def fn(x_np):
        val, grad = vg_fn(jnp.array(x_np, dtype=jnp.float64), *static_args)
        g = np.array(grad, dtype=np.float64)
        if verbose and cc[0] < 3:
            print(f'    [eval {cc[0]}] loss={float(val):.5f}  |g|={np.linalg.norm(g):.3e}', flush=True)
        cc[0] += 1
        return float(val), g
    return fn

def rmse_np(p, o):  return float(np.sqrt(np.mean((p-o)**2)))
def bc_np(p, o):    return 0.5*float(np.sum(np.abs(p-o)))


# ── LOO-CV ─────────────────────────────────────────────────────────────────────

def run_loo(phi_all, patients, args):
    n_p  = len(patients)
    mask = (phi_all.sum(axis=2) > 1e-12).astype(np.float64)

    simulate_single = make_ift_simulate(
        n_sp=N_SP, n_steps=args.n_steps, dt=args.dt,
        c_const=25.0, alpha_const=100.0,
    )
    vg_A, vg_b, vg_bt, fwd_batch, fwd_single = make_loss_fns(
        simulate_single,
        lam_A=args.lambda_a, lam_A_l1=args.lambda_a_l1,
        lam_b=args.lambda_b, alpha_clr=args.alpha_clr,
    )

    # Compile
    n_tr0  = n_p - 1
    _phi0  = jnp.ones((n_tr0, 3, N_SP), dtype=jnp.float64) / N_SP
    _mask0 = jnp.ones((n_tr0, 3), dtype=jnp.float64)
    _b0    = jnp.zeros((n_tr0, N_SP), dtype=jnp.float64)
    _A0    = jnp.array(default_A_upper(), dtype=jnp.float64)
    _phi1  = jnp.ones((3, N_SP), dtype=jnp.float64) / N_SP
    _mask1 = jnp.ones(3, dtype=jnp.float64)

    print(f'α_CLR={args.alpha_clr}  λA={args.lambda_a}  λA_L1={args.lambda_a_l1}  '
          f'λb={args.lambda_b}  maxfun={args.maxfun}', flush=True)
    print('Compiling...', flush=True)
    t0 = time.time()
    _ = vg_A(_A0, _b0, _phi0, _mask0)
    _ = vg_b(_b0.ravel(), _A0, _phi0, _mask0)
    _ = vg_bt(jnp.zeros(N_SP, dtype=jnp.float64), _A0, _phi1, _mask1)
    _ = fwd_batch(_A0, _b0, _phi0[:, 0, :])
    print(f'  compiled: {time.time()-t0:.1f}s', flush=True)

    # Warm-start A
    warm_A = None
    for wp in [_here/'results'/'dieckow_cr'/'fit_guild_hamilton.json',
               _here/'results'/'dieckow_cr'/'fit_guild_hamilton_masked.json']:
        if not wp.exists(): continue
        d = json.load(open(wp)); A0 = np.array(d['A'])
        if A0.shape[0] != N_SP: continue
        warm_A = full_to_A_upper((A0 + A0.T) / 2.0)
        print(f'Warm A from {wp.name}', flush=True); break
    if warm_A is None:
        warm_A = default_A_upper()

    a_bounds = [(-np.inf, 0.0) if k in _DIAG_IDX else (-np.inf, np.inf) for k in range(N_A)]
    results  = []

    for fold in range(args.fold_start, args.fold_end):
        test_pat  = patients[fold]
        print(f'\n{"="*60}\nFold {fold+1}/{n_p}  leave-out: {test_pat}', flush=True)
        t_fold = time.time()

        train_idx    = [i for i, p in enumerate(patients) if p != test_pat]
        test_idx     = patients.index(test_pat)
        n_train      = len(train_idx)
        phi_train_j  = jnp.array(phi_all[np.array(train_idx)], dtype=jnp.float64)
        mask_train_j = jnp.array(mask[np.array(train_idx)],    dtype=jnp.float64)
        phi_test_j   = jnp.array(phi_all[test_idx],            dtype=jnp.float64)
        mask_test_j  = jnp.array(mask[test_idx],               dtype=jnp.float64)
        verbose      = (fold == args.fold_start)

        # Step 1: A
        res_A = minimize(
            _jax_to_scipy(vg_A, jnp.zeros((n_train, N_SP), dtype=jnp.float64),
                          phi_train_j, mask_train_j, verbose=verbose),
            x0=np.array(warm_A), method='L-BFGS-B', jac=True, bounds=a_bounds,
            options=dict(maxiter=args.maxiter, maxfun=args.maxfun, ftol=1e-12, gtol=1e-9),
        )
        A_opt = jnp.array(_clip_diag_np(res_A.x), dtype=jnp.float64)
        print(f'  Step1 A: loss={res_A.fun:.5f}  iters={res_A.nit}  ({time.time()-t_fold:.1f}s)', flush=True)

        # Step 2: b_train
        res_b = minimize(
            _jax_to_scipy(vg_b, A_opt, phi_train_j, mask_train_j, verbose=verbose),
            x0=np.zeros(n_train * N_SP), method='L-BFGS-B', jac=True,
            options=dict(maxiter=args.maxiter, maxfun=args.maxfun, ftol=1e-12, gtol=1e-9),
        )
        b_opt = jnp.array(res_b.x.reshape(n_train, N_SP), dtype=jnp.float64)
        print(f'  Step2 b: loss={res_b.fun:.5f}  iters={res_b.nit}  ({time.time()-t_fold:.1f}s)', flush=True)

        # Train RMSE
        pw2_tr    = np.array(fwd_batch(A_opt, b_opt, phi_train_j[:, 0, :]))
        pw3_tr    = np.array(fwd_batch(A_opt, b_opt, jnp.array(pw2_tr)))
        phi_tr_np = np.array(phi_train_j); mtr = np.array(mask_train_j)
        sq  = np.sum(mtr[:,1,None]*(pw2_tr-phi_tr_np[:,1,:])**2)
        sq += np.sum(mtr[:,2,None]*(pw3_tr-phi_tr_np[:,2,:])**2)
        cnt = np.sum(mtr[:,1]+mtr[:,2]) * N_SP
        print(f'  Train RMSE: {np.sqrt(sq/cnt):.5f}', flush=True)

        # Step 3: b_test
        res_bt = minimize(
            _jax_to_scipy(vg_bt, A_opt, phi_test_j, mask_test_j, verbose=verbose),
            x0=np.full(N_SP, 0.1), method='L-BFGS-B', jac=True,
            options=dict(maxiter=500, maxfun=args.maxfun, ftol=1e-12, gtol=1e-9),
        )
        b_test = jnp.array(res_bt.x, dtype=jnp.float64)
        print(f'  Step3 bt: loss={res_bt.fun:.5f}  iters={res_bt.nit}  ({time.time()-t_fold:.1f}s)', flush=True)

        # Predict W3
        pw2_pred   = np.array(fwd_single(A_opt, b_test, phi_test_j[0]))
        pw3_pred   = np.array(fwd_single(A_opt, b_test, jnp.array(pw2_pred)))
        phi_W3_obs = np.array(phi_test_j[2])
        m3         = float(mask_test_j[2])
        loo_rmse   = rmse_np(pw3_pred, phi_W3_obs) if m3 > 0 else float('nan')
        loo_bc     = bc_np(pw3_pred,   phi_W3_obs) if m3 > 0 else float('nan')

        print(f'  LOO RMSE={loo_rmse:.5f}  BC={loo_bc:.5f}  ({time.time()-t_fold:.1f}s)', flush=True)
        print(f'  W3 pred: {pw3_pred.round(3)}', flush=True)
        print(f'  W3 obs:  {phi_W3_obs.round(3)}', flush=True)
        results.append({'patient': test_pat, 'fold': fold,
                        'rmse': loo_rmse, 'bc': loo_bc})

    valid_r = [r['rmse'] for r in results if not np.isnan(r['rmse'])]
    print(f'\n{"="*60}', flush=True)
    print(f'NSP IFT v7 α={args.alpha_clr} (folds {args.fold_start}-{args.fold_end-1}): '
          f'RMSE={np.mean(valid_r):.5f}  (v2=0.0906, gLV=0.045)', flush=True)
    for r in results:
        print(f"  {r['patient']}: RMSE={r['rmse']:.5f}  BC={r['bc']:.5f}", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi-npy',     default=str(_here/'results'/'dieckow_otu'/'phi_guild.npy'))
    ap.add_argument('--out-json',    default=None)
    ap.add_argument('--n-steps',     type=int,   default=100)
    ap.add_argument('--dt',          type=float, default=0.01)
    ap.add_argument('--maxiter',     type=int,   default=500)
    ap.add_argument('--maxfun',      type=int,   default=5000)
    ap.add_argument('--lambda-a',    type=float, default=1e-4)
    ap.add_argument('--lambda-a-l1', type=float, default=1e-4)
    ap.add_argument('--lambda-b',    type=float, default=1e-3)
    ap.add_argument('--alpha-clr',   type=float, default=0.5,
                    help='CLR weight in mixed loss (0=MSE, 1=CLR)')
    ap.add_argument('--fold-start',  type=int,   default=0)
    ap.add_argument('--fold-end',    type=int,   default=None)
    args = ap.parse_args()

    alpha_tag = str(args.alpha_clr).replace('.', 'p')
    tag = f'a{alpha_tag}_folds{args.fold_start}-{(args.fold_end or 10)-1}'
    if args.out_json is None:
        args.out_json = str(_here/'results'/'dieckow_cr'/f'loo_nsp_ift_v7_{tag}.json')

    print(f'JAX: {jax.devices()}', flush=True)
    phi_all  = np.load(args.phi_npy)
    keep     = phi_all[:, 0, :].sum(axis=1) > 1e-12
    phi_all  = phi_all[keep]
    patients = [p for k, p in zip(keep.tolist(), PAT_ALL) if k]
    n_p = len(patients)
    if args.fold_end is None: args.fold_end = n_p
    args.fold_end = min(args.fold_end, n_p)
    print(f'Patients ({n_p}): {patients}', flush=True)

    results = run_loo(phi_all, patients, args)

    out = Path(args.out_json); out.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if out.exists():
        try: existing = json.load(open(out)).get('per_patient', [])
        except: pass
    ran    = {r['fold'] for r in results}
    merged = [e for e in existing if e.get('fold') not in ran] + results
    merged.sort(key=lambda r: r.get('fold', 0))
    all_r  = [r['rmse'] for r in merged if not np.isnan(r.get('rmse', float('nan')))]
    json.dump({
        'loo_rmse_mean': float(np.mean(all_r)) if all_r else float('nan'),
        'per_patient': merged,
        'model': f'NSP IFT v7 (α_CLR={args.alpha_clr}, maxfun={args.maxfun})',
        'alpha_clr': args.alpha_clr, 'maxfun': args.maxfun,
        'guilds': GUILD_ORDER, 'folds_completed': sorted(r['fold'] for r in merged),
    }, open(out, 'w'), indent=2)
    print(f'Saved: {out}', flush=True)


if __name__ == '__main__':
    main()
