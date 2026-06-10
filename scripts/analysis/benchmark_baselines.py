#!/usr/bin/env python3
"""
benchmark_baselines.py — LOO-CV benchmark comparing gLV against ML baselines.

Task: given phi(W1), predict phi(W2) and phi(W3) for a held-out patient.
Same protocol as run_glv_loo.py:
  - Train on 9 patients (18 transitions)
  - Predict W2 from actual W1, then W3 from predicted W2 (chained)
  - RMSE = sqrt((mean_sq_err_W2 + mean_sq_err_W3) / 2)

Baselines implemented:
  persistence  : phi(t+1) = phi(t)
  mean_train   : predict mean composition of training patients
  ridge        : Ridge regression (multioutput), sklearn
  lasso        : Lasso regression (multioutput via wrapper), sklearn
  rf           : Random Forest regressor (multioutput), sklearn

gLV results are loaded from existing LOO fold JSONs (no-prior and +AGORA).

Output:
  results/benchmark_baselines/rmse_table.json
  results/benchmark_baselines/rmse_per_fold.csv

Usage (from repo root):
  python scripts/analysis/benchmark_baselines.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))

import json, csv
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

_here = Path(__file__).resolve().parents[2]
PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
CR_DIR   = _here / 'results' / 'dieckow_cr'
OUT_DIR  = _here / 'results' / 'benchmark_baselines'
OUT_DIR.mkdir(exist_ok=True)

N_FOLDS  = 10
PATIENTS = list('ABCDEFGHKL')


def _loo_rmse_baseline(predict_fn, phi_all, hold):
    """Compute LOO-RMSE for a baseline given a predict function.

    predict_fn(X_train, y_train, phi_test_W1) -> (phi_pred_W2,)
    Then phi_pred_W3 = predict_fn(X_train, y_train, phi_pred_W2).
    RMSE = sqrt((mse_W2 + mse_W3) / 2).
    """
    tr_idx = [i for i in range(N_FOLDS) if i != hold]
    # Build training pairs: X=phi(t), y=phi(t+1)
    phi_tr = phi_all[tr_idx]          # (9, 3, 10)
    X_tr = phi_tr[:, :-1, :].reshape(-1, phi_all.shape[-1])  # (18, 10)
    y_tr = phi_tr[:, 1:,  :].reshape(-1, phi_all.shape[-1])  # (18, 10)

    phi_ho = phi_all[hold]            # (3, 10)
    phi_W1 = phi_ho[0]

    phi_pred_W2 = predict_fn(X_tr, y_tr, phi_W1)
    phi_pred_W3 = predict_fn(X_tr, y_tr, phi_pred_W2)

    mse_W2 = np.mean((phi_pred_W2 - phi_ho[1])**2)
    mse_W3 = np.mean((phi_pred_W3 - phi_ho[2])**2)
    return float(np.sqrt((mse_W2 + mse_W3) / 2))


# ── Baseline factories ────────────────────────────────────────────────────────

def make_persistence():
    """phi(t+1) = phi(t)"""
    def predict(X_tr, y_tr, phi):
        return phi.copy()
    return predict


def make_mean_train():
    """predict the mean of all training targets (regardless of input)"""
    def predict(X_tr, y_tr, phi):
        return y_tr.mean(axis=0)
    return predict


def make_ridge(alpha=1.0):
    def predict(X_tr, y_tr, phi):
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_tr)
        phi_sc = sc.transform(phi.reshape(1, -1))
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(X_sc, y_tr)
        pred = model.predict(phi_sc)[0]
        pred = np.clip(pred, 0, None)
        s = pred.sum()
        return pred / s if s > 1e-12 else phi.copy()
    return predict


def make_lasso(alpha=0.01):
    def predict(X_tr, y_tr, phi):
        sc = StandardScaler()
        X_sc = sc.fit_transform(X_tr)
        phi_sc = sc.transform(phi.reshape(1, -1))
        model = MultiOutputRegressor(Lasso(alpha=alpha, fit_intercept=True, max_iter=5000))
        model.fit(X_sc, y_tr)
        pred = model.predict(phi_sc)[0]
        pred = np.clip(pred, 0, None)
        s = pred.sum()
        return pred / s if s > 1e-12 else phi.copy()
    return predict


def make_rf(n_estimators=200, max_features='sqrt'):
    def predict(X_tr, y_tr, phi):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            random_state=42,
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(phi.reshape(1, -1))[0]
        pred = np.clip(pred, 0, None)
        s = pred.sum()
        return pred / s if s > 1e-12 else phi.copy()
    return predict


# ── Load existing gLV LOO results ────────────────────────────────────────────

def _load_glv_folds(pattern):
    """Load per-fold RMSE from CR_DIR using a filename pattern with {fold}."""
    rmses = []
    for i in range(N_FOLDS):
        p = CR_DIR / pattern.format(fold=i)
        if not p.exists():
            return None
        rmses.append(json.load(open(p))['rmse'])
    return rmses


def load_glv_results():
    results = {}

    noprior = _load_glv_folds('loo_glv_noprior_a0p0_fold{fold}.json')
    if noprior:
        results['gLV (no prior)'] = noprior

    agora = _load_glv_folds('loo_glv_agora_a0p0_fold{fold}.json')
    if agora:
        results['gLV + AGORA'] = agora

    expanded = _load_glv_folds('loo_expanded_a0p0_fold{fold}.json')
    if expanded:
        results['gLV + L1/L2 prior'] = expanded

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    phi_all = np.load(PHI_NPY)   # (10, 3, 10)
    print(f'phi_guild: {phi_all.shape}  (patients × weeks × guilds)')

    baselines = {
        'persistence':  make_persistence(),
        'mean_train':   make_mean_train(),
        'ridge':        make_ridge(alpha=1.0),
        'lasso':        make_lasso(alpha=0.01),
        'random_forest': make_rf(n_estimators=200),
    }

    # Run LOO for each baseline
    bl_results = {}
    for name, fn in baselines.items():
        rmses = []
        for fold in range(N_FOLDS):
            r = _loo_rmse_baseline(fn, phi_all, fold)
            rmses.append(r)
            print(f'  {name:<16s}  fold {fold}  RMSE={r:.4f}')
        bl_results[name] = rmses
        print(f'  {name:<16s}  MEAN={np.mean(rmses):.4f}  STD={np.std(rmses):.4f}\n')

    # Load gLV results
    glv_results = load_glv_results()
    for name, rmses in glv_results.items():
        print(f'  {name:<22s}  MEAN={np.mean(rmses):.4f}  STD={np.std(rmses):.4f}')

    # Combine all
    all_results = {**bl_results, **glv_results}

    # Build summary table
    summary = {}
    for name, rmses in all_results.items():
        summary[name] = {
            'mean_rmse': float(np.mean(rmses)),
            'std_rmse':  float(np.std(rmses)),
            'per_fold':  [float(r) for r in rmses],
        }

    out_json = OUT_DIR / 'rmse_table.json'
    json.dump(summary, open(out_json, 'w'), indent=2)
    print(f'\nSaved: {out_json}')

    # CSV for easy inspection
    out_csv = OUT_DIR / 'rmse_per_fold.csv'
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        methods = list(all_results.keys())
        w.writerow(['fold', 'patient'] + methods)
        for fold in range(N_FOLDS):
            row = [fold, PATIENTS[fold]] + [f'{all_results[m][fold]:.4f}' for m in methods]
            w.writerow(row)
        w.writerow(['MEAN', ''] + [f'{np.mean(all_results[m]):.4f}' for m in methods])
        w.writerow(['STD',  ''] + [f'{np.std(all_results[m]):.4f}'  for m in methods])
    print(f'Saved: {out_csv}')

    print('\n=== Summary (mean ± std LOO-RMSE) ===')
    for name, v in sorted(summary.items(), key=lambda x: x[1]['mean_rmse']):
        print(f'  {name:<24s}  {v["mean_rmse"]:.4f} ± {v["std_rmse"]:.4f}')


if __name__ == '__main__':
    main()
