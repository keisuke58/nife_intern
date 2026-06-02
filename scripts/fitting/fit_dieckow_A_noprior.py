#!/usr/bin/env python3
"""
fit_dieckow_A_noprior.py — Fit gLV on Dieckow 2024 (10 patients × 3 weeks) WITHOUT
the AGORA sign prior, to test whether the prior is data-consistent or data-overwriting.

Mirrors fit_botelho_A.py exactly (same gLV RHS, same scipy L-BFGS-B, same AGORA
reference prior) so the resulting A is directly comparable to the no-prior Botelho A.

Produces three sign-agreement numbers:
  (1) Dieckow(no-prior) A  vs  AGORA sign prior      → is the prior data-consistent?
  (2) Dieckow(no-prior) A  vs  Botelho(no-prior) A   → prior-free cross-dataset test
  (3) Dieckow(no-prior) A  vs  Dieckow(prior, w1.0) A → how much did the prior move A?

Usage:
    python fit_dieckow_A_noprior.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy import stats

_here = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G
from build_net_flow_expanded import build_net_flow_expanded

DATA = _here / 'results' / 'dieckow_otu'
CR   = _here / 'results' / 'dieckow_cr'
OUT  = _here / 'results' / 'botelho_validation'
OUT.mkdir(parents=True, exist_ok=True)

# ── Load Dieckow data ─────────────────────────────────────────────────────────
phi_all = np.load(DATA / 'phi_guild.npy')     # (10, 3, 10)
N_P, N_T = phi_all.shape[:2]
print(f'Dieckow: {N_P} patients × {N_T} timepoints × {N_G} guilds', flush=True)

# ── AGORA reference sign prior (same as everywhere; used ONLY for comparison) ──
net     = build_net_flow_expanded(use_agora=True, verbose=False, agora_weight=1.0)
net_sym = (net + net.T) / 2.0
sp_mat  = np.sign(net_sym)
mask    = (sp_mat != 0) & (~np.eye(N_G, dtype=bool))
LAM     = 1e-4
n_pairs = int(mask.sum() // 2)
print(f'AGORA constrained pairs (for comparison only): {n_pairs}  PRIOR PENALTY = OFF', flush=True)

# ── gLV helpers (identical to fit_botelho_A.py) ───────────────────────────────
def rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)

def integrate_step(phi0, b, A, dt=1.0):
    sol = solve_ivp(rhs, [0, dt], phi0, args=(b, A),
                    method='RK45', rtol=1e-6, atol=1e-8, dense_output=False)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0.copy()

# ── Full-data training objective (NO sign penalty) ────────────────────────────
def train_obj(x):
    A = x[:N_G * N_G].reshape(N_G, N_G)
    b = x[N_G * N_G:].reshape(N_P, N_G)
    sq, cnt = 0.0, 0
    for p in range(N_P):
        phi_pred = phi_all[p, 0].copy()
        for t in range(1, N_T):
            phi_pred = integrate_step(phi_pred, b[p], A)
            sq  += np.sum((phi_pred - phi_all[p, t]) ** 2)
            cnt += N_G
    rmse = np.sqrt(sq / cnt)
    return rmse + LAM * np.sum(A**2)

bounds = [(None, 0.0) if i == j else (None, None)
          for i in range(N_G) for j in range(N_G)]
bounds += [(None, None)] * (N_P * N_G)

rng = np.random.default_rng(0)
A0  = rng.normal(0, 0.01, (N_G, N_G))
np.fill_diagonal(A0, -0.1)
b0  = np.zeros((N_P, N_G))
x0  = np.concatenate([A0.ravel(), b0.ravel()])

print(f'Initial loss = {train_obj(x0):.5f}', flush=True)
t0 = time.time(); cnt = [0]

def fg(x):
    cnt[0] += 1
    v = train_obj(x)
    if cnt[0] % 100 == 1:
        print(f'  iter {cnt[0]:4d}  loss={v:.5f}  t={time.time()-t0:.0f}s', flush=True)
    return v

res = minimize(fg, x0, method='L-BFGS-B', bounds=bounds,
               options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-10, 'gtol': 1e-7})
print(f'\n{res.message}  iters={cnt[0]}  loss={res.fun:.5f}', flush=True)

A_dk_np = res.x[:N_G * N_G].reshape(N_G, N_G)   # Dieckow no-prior A
b_dk_np = res.x[N_G * N_G:].reshape(N_P, N_G)

# ── Comparison matrices ───────────────────────────────────────────────────────
A_prior = np.array(json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))['A'])
A_bot_np = np.array(json.load(open(OUT / 'botelho_A_comparison_noprior.json'))['A_botelho'])

pairs_all = [(i, j) for i in range(N_G) for j in range(i+1, N_G)]
pairs_con = [(i, j) for i, j in pairs_all if mask[i, j]]

def sign_agree(pairs, A1, A2, label=''):
    matches = [(np.sign(A1[i, j]) == np.sign(A2[i, j])) for i, j in pairs]
    n = len(matches); m = int(sum(matches))
    p = stats.binomtest(m, n, 0.5, alternative='greater').pvalue if n > 0 else 1.0
    print(f'  {label}: {m}/{n} = {m/n:.1%}  p={p:.4f}')
    return m, n

def sign_agree_vs_prior(pairs, A, label=''):
    # compare A's sign to the AGORA prior sign sp_mat on constrained pairs
    matches = [(np.sign(A[i, j]) == sp_mat[i, j]) for i, j in pairs]
    n = len(matches); m = int(sum(matches))
    p = stats.binomtest(m, n, 0.5, alternative='greater').pvalue if n > 0 else 1.0
    print(f'  {label}: {m}/{n} = {m/n:.1%}  p={p:.4f}')
    return m, n

print('\n── (1) Dieckow(no-prior) A  vs  AGORA sign prior ───────────────')
print('     [does the data independently agree with the prior?]')
m1c, n1c = sign_agree_vs_prior(pairs_con, A_dk_np, f'Constrained ({n_pairs} pairs)')

print('\n── (2) Dieckow(no-prior)  vs  Botelho(no-prior)  [PRIOR-FREE CROSS-DATASET] ──')
m2a, n2a = sign_agree(pairs_all, A_dk_np, A_bot_np, 'All 45 pairs')
m2c, n2c = sign_agree(pairs_con, A_dk_np, A_bot_np, f'Constrained ({n_pairs})')
strong2 = [(i, j) for i, j in pairs_all
           if abs(A_dk_np[i, j]) > 0.10 and abs(A_bot_np[i, j]) > 0.10]
if strong2:
    m2s, n2s = sign_agree(strong2, A_dk_np, A_bot_np, f'Strong in both (|A|>0.1, n={len(strong2)})')

print('\n── (3) Dieckow(no-prior)  vs  Dieckow(prior w1.0)  [how far did prior move A?] ──')
m3a, n3a = sign_agree(pairs_all, A_dk_np, A_prior, 'All 45 pairs')
m3c, n3c = sign_agree(pairs_con, A_dk_np, A_prior, f'Constrained ({n_pairs})')

# in-sample RMSE
sq, cnt2 = 0.0, 0
for p in range(N_P):
    phi_pred = phi_all[p, 0].copy()
    for t in range(1, N_T):
        phi_pred = integrate_step(phi_pred, b_dk_np[p], A_dk_np)
        sq += np.sum((phi_pred - phi_all[p, t])**2); cnt2 += N_G
rmse = float(np.sqrt(sq / cnt2))
print(f'\nDieckow(no-prior) in-sample RMSE: {rmse:.4f}')

result = {
    'A_dieckow_noprior': A_dk_np.tolist(),
    'dieckow_noprior_rmse': rmse,
    'sign_agreement': {
        'vs_agora_prior_constrained': {'agree': m1c, 'n': n1c, 'pct': round(m1c/n1c*100, 1)},
        'vs_botelho_noprior_all':     {'agree': m2a, 'n': n2a, 'pct': round(m2a/n2a*100, 1)},
        'vs_botelho_noprior_constrained': {'agree': m2c, 'n': n2c, 'pct': round(m2c/n2c*100, 1)},
        'vs_dieckow_prior_all':       {'agree': m3a, 'n': n3a, 'pct': round(m3a/n3a*100, 1)},
        'vs_dieckow_prior_constrained': {'agree': m3c, 'n': n3c, 'pct': round(m3c/n3c*100, 1)},
    },
    'guild_order': GUILD_ORDER,
}
json.dump(result, open(OUT / 'dieckow_A_noprior_comparison.json', 'w'), indent=2)
print(f'Saved → {OUT}/dieckow_A_noprior_comparison.json', flush=True)
