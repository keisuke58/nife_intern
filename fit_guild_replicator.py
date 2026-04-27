#!/usr/bin/env python3
"""
Fit class-level replicator model to Dieckow phi_guild.npy.
Output: results/dieckow_cr/fit_guild.json
"""

import json, sys, time, argparse
import numpy as np
from scipy.optimize import minimize
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from guild_replicator_dieckow import (
    N_G, GUILD_ORDER, default_A, pack, unpack, conet_edges_to_mask, predict_trajectory,
)

RESULTS_DIR = Path(__file__).parent / 'results' / 'dieckow_cr'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PHI_NPY    = Path(__file__).parent / 'results' / 'dieckow_otu' / 'phi_guild.npy'
LAMBDA_REG = 1e-4
PATIENTS   = list('ABCDEFGHKL')

_call_count = [0]
_t0         = [0.0]

def rmse_guild_masked(A, b_all, phi_obs, present_mask):
    n_p = phi_obs.shape[0]
    sq_sum = 0.0
    count = 0
    for i in range(n_p):
        phi2_pred, phi3_pred = predict_trajectory(phi_obs[i, 0], b_all[i], A)
        if bool(present_mask[i, 1]):
            sq_sum += float(np.sum((phi2_pred - phi_obs[i, 1]) ** 2))
            count += N_G
        if bool(present_mask[i, 2]):
            sq_sum += float(np.sum((phi3_pred - phi_obs[i, 2]) ** 2))
            count += N_G
    return np.sqrt(sq_sum / count) if count > 0 else np.nan


def make_loss(phi_obs, A_sign_prior=None, sign_lambda=0.0):
    n_p = phi_obs.shape[0]
    def loss(theta):
        A, b_all = unpack(theta, n_p)
        r   = rmse_guild_masked(A, b_all, phi_obs, loss.present_mask)
        reg = LAMBDA_REG * np.sum(A**2)
        val = r + reg
        if A_sign_prior is not None and sign_lambda > 0:
            s = np.asarray(A_sign_prior, dtype=float)
            mismatch = np.maximum(0.0, -s * A)
            val = val + float(sign_lambda) * float(np.sum(mismatch ** 2))
        _call_count[0] += 1
        if _call_count[0] % 20 == 0:
            elapsed = time.time() - _t0[0]
            print(f'  call {_call_count[0]:4d}  loss={val:.5f}  ({elapsed:.1f}s)', flush=True)
        return val
    loss.present_mask = np.ones((n_p, 3), dtype=bool)
    return loss


def make_bounds(n_p, A_mask=None, A_sign_prior=None):
    # A: diagonal ≤ 0 (self-limitation), off-diagonal unconstrained
    a_bounds = []
    for i in range(N_G):
        for j in range(N_G):
            if i == j:
                a_bounds.append((None, 0.0))
            else:
                if A_mask is not None and A_mask[i, j] == 0:
                    a_bounds.append((0.0, 0.0))
                else:
                    if A_sign_prior is not None:
                        s = int(A_sign_prior[i, j])
                        if s > 0:
                            a_bounds.append((0.0, None))
                        elif s < 0:
                            a_bounds.append((None, 0.0))
                        else:
                            a_bounds.append((None, None))
                    else:
                        a_bounds.append((None, None))
    b_bounds = [(None, None)] * (n_p * N_G)
    return a_bounds + b_bounds


def run_one(x0, phi_obs, bounds, *, options, A_sign_prior=None, sign_lambda=0.0, label=''):
    loss_fn = make_loss(phi_obs, A_sign_prior=A_sign_prior, sign_lambda=sign_lambda)
    if hasattr(run_one, 'present_mask'):
        loss_fn.present_mask = run_one.present_mask
    result  = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                       options=options)
    n_p = phi_obs.shape[0]
    A, b = unpack(result.x, n_p)
    rmse = rmse_guild_masked(A, b, phi_obs, loss_fn.present_mask)
    print(f'  [{label}] RMSE={rmse:.5f}  calls={_call_count[0]}  {result.message[:60]}',
          flush=True)
    return rmse, result.x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi-npy', default=str(PHI_NPY))
    ap.add_argument('--out-json', default=str(RESULTS_DIR / 'fit_guild.json'))
    ap.add_argument('--warm-start', default=str(RESULTS_DIR / 'fit_guild.json'))
    ap.add_argument('--restarts', type=int, default=4)
    ap.add_argument('--maxiter', type=int, default=1200)
    ap.add_argument('--maxfun', type=int, default=60_000)
    ap.add_argument('--ftol', type=float, default=1e-11)
    ap.add_argument('--gtol', type=float, default=1e-8)
    ap.add_argument('--maxls', type=int, default=60)
    ap.add_argument('--conet-edges', default=None,
                    help='Cytoscape/CoNet edge table (TSV/CSV) with guild names.')
    ap.add_argument('--mask-npy', default=None,
                    help='Optional precomputed A mask .npy (shape N×N, 0/1).')
    ap.add_argument('--use-sign-prior', action='store_true',
                    help='If CoNet sign is available, constrain A_ij sign accordingly.')
    ap.add_argument('--sign-penalty-lambda', type=float, default=0.0,
                    help='Soft penalty weight for sign mismatch (requires CoNet sign).')
    args = ap.parse_args()

    print('Loading phi_guild...', flush=True)
    phi_obs = np.load(args.phi_npy)
    if phi_obs.ndim != 3 or phi_obs.shape[1] != 3:
        raise ValueError(f'Expected phi array shape (n_patients, 3, n_guilds), got {phi_obs.shape}')
    if phi_obs.shape[2] != N_G:
        raise ValueError(f'phi has {phi_obs.shape[2]} guilds but model expects N_G={N_G}.')

    present = phi_obs.sum(axis=2) > 1e-9
    keep = present[:, 0]
    keep_idx = np.flatnonzero(keep)
    phi_obs = phi_obs[keep_idx]
    present = present[keep_idx]
    n_p = phi_obs.shape[0]
    patients_used = [PATIENTS[i] for i in keep_idx] if len(PATIENTS) >= (keep_idx.max() + 1 if keep_idx.size else 0) else [str(i) for i in keep_idx]
    print(f'  {n_p} patients × {phi_obs.shape[1]} weeks × {phi_obs.shape[2]} guilds',
          flush=True)
    print(f'  observed_terms(W2+W3)={int(present[:,1:].sum())}', flush=True)

    A_mask = None
    A_sign = None
    if args.mask_npy is not None:
        A_mask = np.array(np.load(args.mask_npy), dtype=np.int8)
    if args.conet_edges is not None:
        A_mask, A_sign = conet_edges_to_mask(args.conet_edges, undirected=True)
        np.save(RESULTS_DIR / 'conet_guild_A_mask.npy', A_mask)
        np.save(RESULTS_DIR / 'conet_guild_A_sign.npy', A_sign)
        print(f'Loaded CoNet edges → mask: free off-diagonal={int(A_mask.sum())-N_G}', flush=True)

    A_sign_eff = A_sign if (A_sign is not None and (args.use_sign_prior or args.sign_penalty_lambda > 0)) else None
    bounds = make_bounds(n_p, A_mask=A_mask, A_sign_prior=(A_sign_eff if args.use_sign_prior else None))
    opt = dict(maxiter=int(args.maxiter), maxfun=int(args.maxfun),
               ftol=float(args.ftol), gtol=float(args.gtol), maxls=int(args.maxls))

    # --- warm start from previous best if available ---
    starts = []
    warm = Path(args.warm_start)
    if warm.exists():
        prev = json.load(open(warm))
        A0 = np.array(prev.get('A', []), dtype=float)
        b0 = np.array(prev.get('b_all', []), dtype=float)
        if A0.shape == (N_G, N_G) and b0.ndim == 2 and b0.shape[1] == N_G:
            if b0.shape[0] == n_p:
                A_warm = A0
                b_warm = b0
            else:
                prev_pat = prev.get('patients')
                if isinstance(prev_pat, list) and len(prev_pat) == b0.shape[0]:
                    idx_map = {p: i for i, p in enumerate(prev_pat)}
                    rows = [idx_map.get(p) for p in patients_used]
                    if all(r is not None for r in rows):
                        A_warm = A0
                        b_warm = b0[rows]
                    else:
                        A_warm = None
                else:
                    A_warm = None
            if A_warm is not None:
                if A_mask is not None:
                    A_warm = A_warm * A_mask
                np.fill_diagonal(A_warm, np.minimum(np.diag(A_warm), 0.0))
                x_warm = pack(A_warm, np.array(b_warm))
                starts.append(('warm', x_warm))
                print(f'Using warm start from {warm.name}', flush=True)

    # --- random restarts ---
    rng = np.random.default_rng(42)
    for i in range(int(args.restarts)):
        A_rand = default_A() + rng.normal(0, 0.02, (N_G, N_G))
        if A_mask is not None:
            A_rand = A_rand * A_mask
        np.fill_diagonal(A_rand, np.minimum(A_rand.diagonal(), 0))
        b_rand = rng.uniform(0.05, 0.3, (n_p, N_G))
        starts.append((f'rand{i}', pack(A_rand, b_rand)))

    _t0[0] = time.time()
    best_rmse, best_x = np.inf, None
    for label, x0 in starts:
        _call_count[0] = 0
        run_one.present_mask = present
        rmse, x = run_one(
            x0,
            phi_obs,
            bounds,
            options=opt,
            A_sign_prior=A_sign_eff,
            sign_lambda=float(args.sign_penalty_lambda),
            label=label,
        )
        if rmse < best_rmse:
            best_rmse, best_x = rmse, x

    print(f'\nBest RMSE: {best_rmse:.5f}  (total elapsed {time.time()-_t0[0]:.1f}s)',
          flush=True)
    A_map, b_map = unpack(best_x, n_p)
    rmse_final   = rmse_guild_masked(A_map, b_map, phi_obs, present)
    n_warm = 1 if any(lbl == 'warm' for lbl, _ in starts) else 0
    result_msg   = f'Multi-start best (warm={n_warm}, random={int(args.restarts)})'

    print(f'\nFinal RMSE: {rmse_final:.5f}', flush=True)
    print('\nEffective A matrix (rows=target, cols=source):', flush=True)
    header = '         ' + ''.join(f'{g[:4]:>8s}' for g in GUILD_ORDER)
    print(header, flush=True)
    for i, row_name in enumerate(GUILD_ORDER):
        vals = ''.join(f'{A_map[i,j]:8.3f}' for j in range(N_G))
        print(f'  {row_name[:8]:8s} {vals}', flush=True)

    out = Path(args.out_json)
    json.dump(dict(
        A=A_map.tolist(), b_all=b_map.tolist(),
        rmse=rmse_final, guilds=GUILD_ORDER,
        patients=patients_used, success=True,
        phi_npy=str(args.phi_npy),
        observed_terms=int(present[:, 1:].sum()),
        message=result_msg, n_calls=_call_count[0],
        lambda_reg=LAMBDA_REG,
    ), open(out, 'w'), indent=2)
    print(f'\nSaved: {out}', flush=True)


if __name__ == '__main__':
    main()
