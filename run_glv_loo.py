#!/usr/bin/env python3
"""
run_glv_loo.py — LOO-CV for gLV replicator model with expanded sign prior.

For each held-out patient:
  1. Train A on remaining 9 patients (replicator ODE + sign prior from net_flow_glv)
  2. Fit b for held-out patient (A fixed)
  3. Predict weeks 2,3 → LOO-CV RMSE

Usage:
    python run_glv_loo.py --hold 0 --alpha 0.5
    python run_glv_loo.py --hold 3 --alpha 0.0
"""
import argparse, json, sys, time
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

parser = argparse.ArgumentParser()
parser.add_argument('--hold',     type=int,   required=True)
parser.add_argument('--alpha',    type=float, default=0.0,
                    help='competition_weight for net_flow_glv (0=cross-feeding only)')
parser.add_argument('--use-agora', action='store_true',
                    help='include AGORA2 FBA layer (L3) in sign prior')
parser.add_argument('--agora-medium', type=str, default='v1',
                    choices=['v1', 'v2', 'micom'],
                    help='AGORA mode: v1=blood-plasma pFBA, v2=saliva pFBA, micom=community FBA')
parser.add_argument('--micom-fraction', type=float, default=0.5,
                    help='MICOM cooperative tradeoff fraction tau (default 0.5, Diener 2020)')
parser.add_argument('--no-prior',  action='store_true',
                    help='disable sign prior entirely (pure data fit, baseline)')
parser.add_argument('--fit-file', type=str,
                    default='fit_glv_hamilton_kegg_expanded_agora_w0p5.json',
                    help='JSON warm-start file (A, b_all)')
parser.add_argument('--tag', type=str, default='',
                    help='output filename tag')
parser.add_argument('--maxiter', type=int, default=2000,
                    help='L-BFGS-B max iterations for A training (default 2000)')
args = parser.parse_args()

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

from guild_replicator_dieckow import GUILD_ORDER, N_G
from build_net_flow_expanded import net_flow_glv

PATIENTS = list('ABCDEFGHKL')
PHI_NPY  = _here / 'results' / 'dieckow_otu' / 'phi_guild.npy'
OUT_DIR  = _here / 'results' / 'dieckow_cr'
SIGMA    = 0.15
LAM      = 1e-4

# Sign prior (directed, not symmetrized)
net_sym = net_flow_glv(use_agora=args.use_agora, competition_weight=args.alpha,
                       agora_medium=args.agora_medium,
                       micom_fraction=args.micom_fraction)
sp_mat  = np.sign(net_sym)
mask_np = (sp_mat != 0) & (~np.eye(N_G, dtype=bool))
print(f'net_flow_glv  alpha={args.alpha}  agora={args.use_agora}  '
      f'medium={args.agora_medium}  fraction={args.micom_fraction}  '
      f'pos={int((net_sym>0).sum())}  neg={int((net_sym<0).sum())}', flush=True)

# Data
phi_all = np.load(PHI_NPY)   # (10,3,10)

# Warm start
_fit_d = json.load(open(OUT_DIR / args.fit_file))
A_warm = np.array(_fit_d['A'])     # (10,10)
b_warm = np.array(_fit_d['b_all']) # (10,10)

hold   = args.hold
tr_idx = [i for i in range(10) if i != hold]
phi_tr = phi_all[tr_idx]   # (9,3,10)
phi_ho = phi_all[hold]     # (3,10)
n_tr   = len(tr_idx)

print(f'LOO fold {hold}: held={PATIENTS[hold]}  train={[PATIENTS[i] for i in tr_idx]}',
      flush=True)


# ── gLV helpers ───────────────────────────────────────────────────────────────

def _replicator_rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)


def _integrate_step(phi0, b, A):
    sol = solve_ivp(_replicator_rhs, [0, 1.0], phi0, args=(b, A),
                    method='RK45', rtol=1e-6, atol=1e-8, dense_output=False)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0.copy()


def _sign_penalty(A):
    pen = 0.0
    for i in range(N_G):
        for j in range(N_G):
            if mask_np[i, j]:
                w = abs(net_sym[i, j])
                v = max(0.0, -sp_mat[i, j] * A[i, j])
                pen += w * v * v / (2 * SIGMA * SIGMA)
    return pen


# ── Training objective ────────────────────────────────────────────────────────

def _train_obj(x_flat):
    A = x_flat[:N_G * N_G].reshape(N_G, N_G)
    b = x_flat[N_G * N_G:].reshape(n_tr, N_G)
    sq, cnt = 0.0, 0
    for k in range(n_tr):
        phi2 = _integrate_step(phi_tr[k, 0], b[k], A)
        phi3 = _integrate_step(phi2,          b[k], A)
        sq  += np.sum((phi2 - phi_tr[k, 1])**2) + np.sum((phi3 - phi_tr[k, 2])**2)
        cnt += 2 * N_G
    rmse = np.sqrt(sq / cnt)
    pen  = 0.0 if args.no_prior else _sign_penalty(A)
    return rmse + pen + LAM * np.sum(A**2)


def _bounds_train():
    ba = [(None, 0.0) if i == j else (None, None)
          for i in range(N_G) for j in range(N_G)]
    bb = [(None, None)] * (n_tr * N_G)
    return ba + bb


# ── Train A + b on 9-patient set ──────────────────────────────────────────────

A_tr_warm = A_warm.copy()
b_tr_warm = b_warm[tr_idx].copy()
x0_train  = np.concatenate([A_tr_warm.ravel(), b_tr_warm.ravel()])

init_loss = _train_obj(x0_train)
print(f'Initial loss={init_loss:.5f}', flush=True)

call_count = [0]; t0 = time.time()

def _fg_train(x):
    call_count[0] += 1
    loss = _train_obj(x)
    if call_count[0] % 50 == 1:
        print(f'  iter {call_count[0]:4d}  loss={float(loss):.5f}  ({time.time()-t0:.1f}s)',
              flush=True)
    return loss

res = minimize(_fg_train, x0_train, method='L-BFGS-B', bounds=_bounds_train(),
               options={'maxiter': args.maxiter, 'ftol': 1e-10, 'gtol': 1e-7})
print(f'  {res.message}  iters={call_count[0]}  loss={res.fun:.5f}', flush=True)

A_opt = res.x[:N_G * N_G].reshape(N_G, N_G)
b_tr_opt = res.x[N_G * N_G:].reshape(n_tr, N_G)


# ── Fit b for held-out patient ────────────────────────────────────────────────

def _ho_obj(b_p):
    phi2 = _integrate_step(phi_ho[0], b_p, A_opt)
    phi3 = _integrate_step(phi2,       b_p, A_opt)
    sq = np.sum((phi2 - phi_ho[1])**2) + np.sum((phi3 - phi_ho[2])**2)
    return float(np.sqrt(sq / (2 * N_G)))

res_b = minimize(_ho_obj, b_warm[hold].copy(), method='L-BFGS-B',
                 options={'maxiter': max(500, args.maxiter // 4), 'ftol': 1e-12, 'gtol': 1e-7})
b_ho_opt = res_b.x


# ── Evaluate ──────────────────────────────────────────────────────────────────

phi2 = _integrate_step(phi_ho[0], b_ho_opt, A_opt)
phi3 = _integrate_step(phi2,       b_ho_opt, A_opt)
rmse = float(np.sqrt((np.mean((phi2 - phi_ho[1])**2) +
                      np.mean((phi3 - phi_ho[2])**2)) / 2))

n_agree = int(((np.sign(A_opt) == sp_mat) & mask_np).sum())
n_tot   = int(mask_np.sum())
print(f'\nFold {hold} ({PATIENTS[hold]}): LOO-RMSE={rmse:.4f}  SA={n_agree}/{n_tot}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────

alpha_tag   = f'_a{str(args.alpha).replace(".", "p")}'
agora_tag   = '_agora' if args.use_agora else ''
medium_tag  = f'_med{args.agora_medium}' if args.use_agora and args.agora_medium != 'v1' else ''
noprior_tag = '_noprior' if args.no_prior else ''
tag         = f'_{args.tag}' if args.tag else ''
fname = f'loo_glv{noprior_tag}{agora_tag}{medium_tag}{alpha_tag}{tag}_fold{hold}.json'
out = {
    'patient':        PATIENTS[hold],
    'hold_idx':       hold,
    'rmse':           rmse,
    'sign_agreement': f'{n_agree}/{n_tot}',
    'fit_file':       args.fit_file,
    'alpha':          args.alpha,
    'use_agora':      args.use_agora,
    'agora_medium':   args.agora_medium,
    'micom_fraction': args.micom_fraction,
    'no_prior':       args.no_prior,
    'tag':            args.tag,
    'maxiter':        args.maxiter,
    'A':              A_opt.tolist(),
    'b_ho':           b_ho_opt.tolist(),
    'model':          (f'gLV replicator LOO (no_prior={args.no_prior}, alpha={args.alpha}, '
                       f'agora={args.use_agora}, medium={args.agora_medium}, '
                       f'fraction={args.micom_fraction}, {n_tot} constrained elements)'),
}
json.dump(out, open(OUT_DIR / fname, 'w'), indent=2)
print(f'Saved {fname}', flush=True)
