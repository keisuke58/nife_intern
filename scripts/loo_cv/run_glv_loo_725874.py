#!/usr/bin/env python3
"""
run_glv_loo_725874.py — LOO-CV for gLV on PRJNA725874 (periodontitis, 7 timepoints)

For each held-out patient:
  1. Train A on remaining 14 patients (T00→T02→...→T12, 6 steps of 2 months)
  2. Fit b for held-out patient (A fixed)
  3. Predict T02..T12 from T00 → LOO-RMSE and Bray-Curtis

Usage:
    python run_glv_loo_725874.py --hold 0 --alpha 0.0
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import argparse, json, sys, time
import numpy as np
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

parser = argparse.ArgumentParser()
parser.add_argument('--hold',     type=int,   required=True)
parser.add_argument('--alpha',    type=float, default=0.0)
parser.add_argument('--use-agora', action='store_true')
parser.add_argument('--no-prior',  action='store_true')
args = parser.parse_args()

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))

from build_net_flow_expanded import net_flow_glv

DATA    = _here / 'data' / 'prjna725874'
OUT_DIR = _here / 'results' / 'prjna725874'
OUT_DIR.mkdir(parents=True, exist_ok=True)

meta    = json.load(open(DATA / 'metadata.json'))
PATIENTS   = meta['patients']           # ['1','2',...,'15']
TIMEPOINTS = meta['timepoints']         # ['00','02',...,'12']
GUILD_ORDER = meta['guilds']
N_G  = len(GUILD_ORDER)
N_T  = len(TIMEPOINTS)        # 7
SIGMA = 0.15
LAM   = 1e-4

# Sign prior
net_sym = net_flow_glv(use_agora=args.use_agora, competition_weight=args.alpha)
sp_mat  = np.sign(net_sym)
mask_np = (sp_mat != 0) & (~np.eye(N_G, dtype=bool))
print(f'net_flow_glv  alpha={args.alpha}  agora={args.use_agora}  '
      f'constrained={int(mask_np.sum())}', flush=True)

# Data: (N_P, N_T, N_G)
phi_all = np.load(DATA / 'phi_guild.npy')
N_P = len(PATIENTS)
assert phi_all.shape == (N_P, N_T, N_G), f"Shape mismatch: {phi_all.shape}"

hold   = args.hold
tr_idx = [i for i in range(N_P) if i != hold]
phi_tr = phi_all[tr_idx]    # (N_P-1, N_T, N_G)
phi_ho = phi_all[hold]      # (N_T, N_G)
n_tr   = len(tr_idx)

print(f'LOO fold {hold}: held=P{PATIENTS[hold]}  n_train={n_tr}', flush=True)


# ── gLV helpers ───────────────────────────────────────────────────────────────

def _replicator_rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)


def _integrate_step(phi0, b, A):
    sol = solve_ivp(_replicator_rhs, [0, 1.0], phi0, args=(b, A),
                    method='RK45', rtol=1e-6, atol=1e-8)
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
# Integrate N_T-1 steps: T00→T02→...→T12, minimize RMSE vs observed T02..T12

def _train_obj(x_flat):
    A = x_flat[:N_G * N_G].reshape(N_G, N_G)
    b = x_flat[N_G * N_G:].reshape(n_tr, N_G)
    sq, cnt = 0.0, 0
    for k in range(n_tr):
        phi_pred = phi_tr[k, 0].copy()
        for t in range(1, N_T):
            phi_pred = _integrate_step(phi_pred, b[k], A)
            sq += np.sum((phi_pred - phi_tr[k, t]) ** 2)
            cnt += N_G
    rmse = np.sqrt(sq / cnt)
    pen  = 0.0 if args.no_prior else _sign_penalty(A)
    return rmse + pen + LAM * np.sum(A ** 2)


def _bounds_train():
    ba = [(None, 0.0) if i == j else (None, None)
          for i in range(N_G) for j in range(N_G)]
    bb = [(None, None)] * (n_tr * N_G)
    return ba + bb


# ── Warm start: small random A, zero b ───────────────────────────────────────
rng = np.random.default_rng(42 + hold)
A0  = rng.normal(0, 0.01, (N_G, N_G))
np.fill_diagonal(A0, -0.1)
b0  = np.zeros((n_tr, N_G))
x0  = np.concatenate([A0.ravel(), b0.ravel()])

init_loss = _train_obj(x0)
print(f'Initial loss={init_loss:.5f}', flush=True)

call_count = [0]; t0 = time.time()

def _fg_train(x):
    call_count[0] += 1
    loss = _train_obj(x)
    if call_count[0] % 100 == 1:
        print(f'  iter {call_count[0]:4d}  loss={float(loss):.5f}  ({time.time()-t0:.1f}s)',
              flush=True)
    return loss

res = minimize(_fg_train, x0, method='L-BFGS-B', bounds=_bounds_train(),
               options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-7})
print(f'  {res.message}  iters={call_count[0]}  loss={res.fun:.5f}', flush=True)

A_opt    = res.x[:N_G * N_G].reshape(N_G, N_G)
b_tr_opt = res.x[N_G * N_G:].reshape(n_tr, N_G)


# ── Fit b for held-out patient ────────────────────────────────────────────────

def _ho_obj(b_p):
    phi_pred = phi_ho[0].copy()
    sq = 0.0
    for t in range(1, N_T):
        phi_pred = _integrate_step(phi_pred, b_p, A_opt)
        sq += np.sum((phi_pred - phi_ho[t]) ** 2)
    return float(np.sqrt(sq / ((N_T - 1) * N_G)))

res_b = minimize(_ho_obj, np.zeros(N_G), method='L-BFGS-B',
                 options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-7})
b_ho_opt = res_b.x


# ── Evaluate ──────────────────────────────────────────────────────────────────

phi_pred = phi_ho[0].copy()
sq, bc_sum = 0.0, 0.0
pred_traj = [phi_ho[0].copy()]
for t in range(1, N_T):
    phi_pred = _integrate_step(phi_pred, b_ho_opt, A_opt)
    pred_traj.append(phi_pred.copy())
    sq     += np.sum((phi_pred - phi_ho[t]) ** 2)
    bc_sum += 0.5 * np.sum(np.abs(phi_pred - phi_ho[t]))

rmse = float(np.sqrt(sq / ((N_T - 1) * N_G)))
bc   = float(bc_sum / (N_T - 1))

n_agree = int(((np.sign(A_opt) == sp_mat) & mask_np).sum())
n_tot   = int(mask_np.sum())

print(f'\nFold {hold} (P{PATIENTS[hold]}): LOO-RMSE={rmse:.4f}  BC={bc:.4f}  '
      f'SA={n_agree}/{n_tot}', flush=True)


# ── Save ──────────────────────────────────────────────────────────────────────

alpha_tag   = f'_a{str(args.alpha).replace(".", "p")}'
agora_tag   = '_agora' if args.use_agora else ''
noprior_tag = '_noprior' if args.no_prior else ''
fname = f'loo_glv{noprior_tag}{agora_tag}{alpha_tag}_fold{hold}.json'

out = {
    'patient':        PATIENTS[hold],
    'hold_idx':       hold,
    'rmse':           rmse,
    'bc':             bc,
    'sign_agreement': f'{n_agree}/{n_tot}',
    'alpha':          args.alpha,
    'use_agora':      args.use_agora,
    'no_prior':       args.no_prior,
    'n_timepoints':   N_T,
    'dataset':        'PRJNA725874',
    'A':              A_opt.tolist(),
    'b_ho':           b_ho_opt.tolist(),
    'pred_traj':      [p.tolist() for p in pred_traj],
}
json.dump(out, open(OUT_DIR / fname, 'w'), indent=2)
print(f'Saved {OUT_DIR / fname}', flush=True)
