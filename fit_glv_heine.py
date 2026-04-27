#!/usr/bin/env python3
"""
gLV replicator ODE fit to Heine 2025 5-species biofilm data.

Per condition (CS/CH/DS/DH): fit A (5x5 asymmetric) + b (5) to minimise
RMSE of integrated trajectory vs median observed composition.

ODE: dphi_i/dt = phi_i * (sum_j A_ij phi_j + b_i - fbar)
Initial cond: day 1 median, targets: days 3,6,10,15,21.

Output: results/heine2025/fit_glv_heine.json
"""

import json, time, argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from nife.comets.oral_biofilm import metabolic_interaction_prior

DATA_CSV    = Path('/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/experiment_data/fig3_species_distribution_replicates.csv')
RESULTS_DIR = Path('/home/nishioka/IKM_Hiwi/nife/results/heine2025')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DAYS   = [1, 3, 6, 10, 15, 21]
N_SP   = 5
LAMBDA = 1e-3

CONDITIONS = [
    ('Commensal', 'Static',  'CS'),
    ('Commensal', 'HOBIC',   'CH'),
    ('Dysbiotic', 'Static',  'DS'),
    ('Dysbiotic', 'HOBIC',   'DH'),
]
SPECIES = {
    'Commensal': ['S. oralis', 'A. naeslundii', 'V. dispar',  'F. nucleatum', 'P. gingivalis_20709'],
    'Dysbiotic': ['S. oralis', 'A. naeslundii', 'V. parvula', 'F. nucleatum', 'P. gingivalis_W83'],
}
SHORT = ['So', 'An', 'Vd/Vp', 'Fn', 'Pg']


def replicator_rhs(t, phi, A, b):
    phi = np.maximum(phi, 1e-10)
    phi = phi / phi.sum()
    f    = A @ phi + b
    fbar = phi @ f
    return phi * (f - fbar)


def integrate_glv(A, b, phi0, days):
    sol = solve_ivp(replicator_rhs, [days[0], days[-1]], phi0,
                    t_eval=days, args=(A, b), method='RK45',
                    rtol=1e-6, atol=1e-9, max_step=1.0)
    traj = sol.y.T
    traj = np.maximum(traj, 0)
    traj = traj / traj.sum(axis=1, keepdims=True)
    return traj


def rmse_traj(A, b, phi_obs):
    try:
        pred = integrate_glv(A, b, phi_obs[0], DAYS)
        return float(np.sqrt(np.mean((pred - phi_obs) ** 2)))
    except Exception:
        return 1.0


def make_loss(phi_obs):
    def loss(x, sign_prior=None, sign_lambda=0.0):
        A = x[:N_SP * N_SP].reshape(N_SP, N_SP)
        b = x[N_SP * N_SP:]
        r   = rmse_traj(A, b, phi_obs)
        reg = LAMBDA * np.sum(A ** 2)
        if sign_prior is None or sign_lambda <= 0:
            return r + reg
        s = np.asarray(sign_prior, dtype=float)
        s = np.where(np.eye(N_SP, dtype=bool), 0.0, s)
        mismatch = np.maximum(0.0, -s * A)
        pen = float(sign_lambda) * float(np.sum(mismatch ** 2))
        return r + reg + pen
    return loss


def _sign_prior_for_condition_key(cond_key: str) -> np.ndarray:
    prof = metabolic_interaction_prior(cond_key, flux_threshold=1e-7, min_count=1)
    return np.array(prof["prior"]["sign"], dtype=int)


def _mechanistic_metrics(A: np.ndarray, sign_prior_source_target: np.ndarray) -> tuple[float, float]:
    s = np.asarray(sign_prior_source_target, dtype=float)
    mask = (s != 0) & (~np.eye(N_SP, dtype=bool))
    if not np.any(mask):
        return 0.0, float("nan")
    a_src_tgt = A.T
    mismatch = np.maximum(0.0, -s * a_src_tgt)
    mismatch_sum = float(mismatch[mask].sum())
    consistency = float(((s * a_src_tgt) >= 0.0)[mask].mean())
    return mismatch_sum, consistency


def load_phi(df, condition, cultivation):
    sp_list = SPECIES[condition]
    mask    = (df['condition'] == condition) & (df['cultivation'] == cultivation)
    sub     = df[mask]
    phi     = np.zeros((len(DAYS), N_SP))
    for j, sp in enumerate(sp_list):
        sp_sub = sub[sub['species'] == sp]
        for i, day in enumerate(DAYS):
            vals = sp_sub[sp_sub['day'] == day]['distribution_pct'].values
            phi[i, j] = np.median(vals) if len(vals) > 0 else 0.0
    row_sums = phi.sum(axis=1, keepdims=True)
    phi = phi / np.where(row_sums > 0, row_sums, 1.0)
    return phi


def fit_condition(phi_obs, rng, sign_prior=None, sign_lambda=0.0, hard_sign=False, maxiter=3000, nstarts=5):
    base_loss = make_loss(phi_obs)
    sign_prior_A = None
    if sign_prior is not None:
        sign_prior_A = np.asarray(sign_prior, dtype=int).T
    loss_fn = lambda x: base_loss(x, sign_prior=sign_prior_A, sign_lambda=sign_lambda)
    best_val, best_x = np.inf, None
    n = N_SP * N_SP + N_SP

    nstarts = int(max(1, nstarts))
    starts = [np.concatenate([-0.1 * np.eye(N_SP).ravel(), np.full(N_SP, 0.1)])]
    for _ in range(nstarts - 1):
        starts.append(rng.normal(0, 0.15, n))

    bounds = None
    if hard_sign and sign_prior_A is not None:
        bounds = []
        for i in range(N_SP):
            for j in range(N_SP):
                if i == j:
                    bounds.append((None, None))
                    continue
                s = int(sign_prior_A[i][j])
                if s > 0:
                    bounds.append((0.0, None))
                elif s < 0:
                    bounds.append((None, 0.0))
                else:
                    bounds.append((None, None))
        bounds += [(None, None)] * N_SP

    for x0 in starts:
        res = minimize(loss_fn, x0, method='L-BFGS-B', bounds=bounds,
                       options=dict(maxiter=int(maxiter), maxfun=10000, ftol=1e-12, gtol=1e-8))
        if res.fun < best_val:
            best_val, best_x = res.fun, res.x.copy()

    A    = best_x[:N_SP * N_SP].reshape(N_SP, N_SP)
    b    = best_x[N_SP * N_SP:]
    rmse = rmse_traj(A, b, phi_obs)
    return A, b, rmse


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanistic", choices=["none", "soft", "hard", "both"], default="none")
    ap.add_argument("--sign-lambda", type=float, default=1e-2)
    ap.add_argument("--sweep-sign-lambda", type=str, default="")
    ap.add_argument("--maxiter", type=int, default=3000)
    ap.add_argument("--nstarts", type=int, default=5)
    ap.add_argument("--out", type=Path, default=RESULTS_DIR / "fit_glv_heine.json")
    args = ap.parse_args(argv)

    df  = pd.read_csv(DATA_CSV)
    rng = np.random.default_rng(42)
    t0  = time.time()
    print(f'Loaded {len(df)} rows\n')

    if args.sweep_sign_lambda:
        lambdas = [float(x) for x in args.sweep_sign_lambda.split(",") if x.strip()]
        mech = "soft"
    else:
        lambdas = [float(args.sign_lambda)]
        mech = str(args.mechanistic)

    prior_cache: dict[str, np.ndarray] = {}
    def prior_for_label(label: str) -> np.ndarray | None:
        if mech == "none":
            return None
        cond_key = "commensal" if label in ("CS", "CH") else "dysbiotic"
        if cond_key not in prior_cache:
            prior_cache[cond_key] = _sign_prior_for_condition_key(cond_key)
        return prior_cache[cond_key]

    sweep_rows: list[dict] = []
    best_score = np.inf
    best_results: dict | None = None
    best_lambda: float | None = None

    for lam in lambdas:
        results = {}
        ok_all = True
        rmse_mean = 0.0
        for condition, cultivation, label in CONDITIONS:
            phi_obs = load_phi(df, condition, cultivation)
            print(f'── {label} ──  phi_obs: {phi_obs.shape}')
            print(f'  day1: {" ".join(f"{v:.2f}" for v in phi_obs[0])}')

            sign_prior = prior_for_label(label)
            use_soft = mech in ("soft", "both")
            use_hard = mech in ("hard", "both")
            A, b, rmse = fit_condition(
                phi_obs,
                rng,
                sign_prior=sign_prior,
                sign_lambda=(lam if use_soft else 0.0),
                hard_sign=use_hard,
                maxiter=args.maxiter,
                nstarts=args.nstarts,
            )
            mismatch_sum, consistency = (0.0, float("nan"))
            if sign_prior is not None:
                mismatch_sum, consistency = _mechanistic_metrics(A, sign_prior)
                if not (np.isfinite(consistency) and abs(consistency - 1.0) < 1e-12 and mismatch_sum <= 1e-12):
                    ok_all = False

            rmse_mean += float(rmse)
            print(f'  RMSE={rmse:.5f}  ({time.time()-t0:.1f}s)')

            header = '       ' + ''.join(f'{s:>8s}' for s in SHORT)
            print(f'  {header}')
            for i, s in enumerate(SHORT):
                row = ''.join(f'{A[i,j]:8.3f}' for j in range(N_SP))
                print(f'  {s:6s} {row}')

            results[label] = dict(
                A=A.tolist(),
                b=b.tolist(),
                rmse=rmse,
                species=SPECIES[condition],
                condition=condition,
                cultivation=cultivation,
            )
            sweep_rows.append(
                dict(
                    sign_lambda=float(lam),
                    label=label,
                    rmse=float(rmse),
                    mismatch_sum=float(mismatch_sum),
                    consistency=float(consistency),
                )
            )

        rmse_mean /= float(len(CONDITIONS))
        if ok_all and rmse_mean < best_score:
            best_score = rmse_mean
            best_results = results
            best_lambda = float(lam)

    if args.sweep_sign_lambda:
        out_csv = args.out.with_name(args.out.stem + "_signlambda_sweep.csv")
        pd.DataFrame(sweep_rows).to_csv(out_csv, index=False)
        print(f'\nSaved: {out_csv}')

        if best_results is None:
            print("No sign_lambda achieved consistency=1.00 for all labels.")
            return
        json.dump(best_results, open(args.out, 'w'), indent=2)
        print(f'Saved: {args.out}')
        print(f'Best sign_lambda: {best_lambda}')
        print(f'Total: {time.time()-t0:.1f}s')
        return

    json.dump(results, open(args.out, 'w'), indent=2)
    print(f'\nSaved: {args.out}')
    print(f'Total: {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
