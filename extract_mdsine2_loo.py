#!/usr/bin/env python3
"""
extract_mdsine2_loo.py
  Post-processing: read MDSINE2 HDF5 traces → compute LOO RMSE → save JSON.
  Use after MCMC completed but the main script crashed at the pickle step.

Usage:
    /home/nishioka/IKM_Hiwi/.venv_jax/bin/python extract_mdsine2_loo.py
"""
import json, sys
import numpy as np
import h5py
from pathlib import Path
from scipy.optimize import minimize

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
sys.path.insert(0, str(_here.parent / 'Tmcmc202601' / 'data_5species' / 'main'))
from guild_replicator_dieckow import GUILD_ORDER, N_G

PATIENTS = list('ABCDEFGHKL')
OUT_DIR  = _here / 'results' / 'mdsine2'
OTU      = _here / 'results' / 'dieckow_otu'

with open(_here / 'results' / 'dieckow_otu' / 'guild_summary.json') as f:
    gs = json.load(f)
GUILDS = gs['guilds']
obs    = {}
for s in gs['samples']:
    obs[(s['patient'], s['week'])] = np.array([s[g] for g in GUILDS])

def _step(phi0, b, A, n_steps=2000, dt=5e-4):
    """Fast numpy Euler integrator for 1-week replicator step."""
    phi = phi0.copy()
    for _ in range(n_steps):
        f    = b + A @ phi
        mf   = phi @ f
        dphi = phi * (f - mf)
        phi  = phi + dt * dphi
        phi  = np.maximum(phi, 0.0)
        s    = phi.sum()
        if s > 1e-12:
            phi /= s
    return phi

def process_fold(fold):
    trace_file = OUT_DIR / f'fold{fold}' / 'traces.hdf5'
    out_file   = OUT_DIR / f'loo_fold{fold}.json'

    if out_file.exists():
        print(f'fold {fold}: already done, skipping.')
        return

    if not trace_file.exists():
        print(f'fold {fold}: trace not found, skipping.')
        return

    with h5py.File(trace_file, 'r') as hf:
        A_raw    = hf['Interactions object'][:]                     # (n, N_G, N_G) with NaN
        ind      = hf['Cluster interaction indicator parameter'][:]  # (n, N_G, N_G) bool
        b_samples = hf['Growth parameter'][:]                        # (n, N_G)

    # Zero out inactive interactions (NaN where indicator=False)
    A_samples = np.where(ind, A_raw, 0.0)
    n_post    = len(A_samples)
    A_mean    = A_samples.mean(axis=0)
    b_mean    = b_samples.mean(axis=0)
    nz_frac   = ind.mean()
    print(f'  A nonzero fraction: {nz_frac:.4f} (effective A {"≈0" if nz_frac < 0.01 else ""})')
    print(f'fold {fold}: {n_post} posterior samples loaded.')

    hold_pat = PATIENTS[fold]
    phi_ho   = [obs[(hold_pat, wk)] for wk in [1, 2, 3]]

    def _ho_obj(b_p):
        p2 = _step(phi_ho[0], b_p, A_mean)
        p3 = _step(p2,        b_p, A_mean)
        return float(np.sqrt((np.mean((p2 - phi_ho[1])**2) +
                              np.mean((p3 - phi_ho[2])**2)) / 2))

    res_b    = minimize(_ho_obj, b_mean.copy(), method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-12})
    b_ho_opt = res_b.x

    p2   = _step(phi_ho[0], b_ho_opt, A_mean)
    p3   = _step(p2,        b_ho_opt, A_mean)
    rmse = float(np.sqrt((np.mean((p2 - phi_ho[1])**2) +
                          np.mean((p3 - phi_ho[2])**2)) / 2))

    print(f'  fold {fold} ({hold_pat}): LOO-RMSE={rmse:.4f}')

    out = {
        'patient':   hold_pat,
        'hold_idx':  fold,
        'rmse':      rmse,
        'A_mean':    A_mean.tolist(),
        'b_ho':      b_ho_opt.tolist(),
        'n_samples': n_post,
    }
    json.dump(out, open(out_file, 'w'), indent=2)
    print(f'  Saved {out_file.name}')

for fold in range(10):
    try:
        process_fold(fold)
    except Exception as e:
        print(f'  fold {fold}: ERROR — {e}')

# Print summary if all done
results = []
for fold in range(10):
    fp = OUT_DIR / f'loo_fold{fold}.json'
    if fp.exists():
        d = json.loads(fp.read_text())
        results.append(d)

if results:
    rmses = [d['rmse'] for d in results]
    pats  = [d['patient'] for d in results]
    print(f'\n=== MDSINE2 LOO Summary ({len(results)}/10 folds) ===')
    for d in results:
        print(f'  fold {d["hold_idx"]} ({d["patient"]}): RMSE={d["rmse"]:.4f}')
    print(f'  Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}')
