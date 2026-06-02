#!/usr/bin/env python3
"""
fit_botelho_A.py — Fit gLV on Botelho 2021 (PRJNA725874, all 15 patients)
and compare resulting A matrix sign-by-sign with Dieckow AGORA W=1.0 A.

This is the key cross-dataset validation:
  If Botelho A and Dieckow A agree on signs > chance,
  it means both independent datasets support the same ecological interaction structure.

Usage:
    python fit_botelho_A.py [--no-prior]
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('--no-prior', action='store_true')
args = parser.parse_args()

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))
from guild_replicator_dieckow import GUILD_ORDER, N_G
from build_net_flow_expanded import build_net_flow_expanded

DATA   = _here / 'data'   / 'prjna725874'
CR     = _here / 'results' / 'dieckow_cr'
OUT    = _here / 'results' / 'botelho_validation'
OUT.mkdir(parents=True, exist_ok=True)

# ── Load Botelho data ─────────────────────────────────────────────────────────
phi_all = np.load(DATA / 'phi_guild.npy')     # (15, 7, 10)
meta    = json.load(open(DATA / 'metadata.json'))
N_P, N_T = phi_all.shape[:2]
print(f'Botelho: {N_P} patients × {N_T} timepoints × {N_G} guilds', flush=True)

# ── Sign prior (same as Dieckow) ──────────────────────────────────────────────
net     = build_net_flow_expanded(use_agora=True, verbose=False, agora_weight=1.0)
net_sym = (net + net.T) / 2.0
sp_mat  = np.sign(net_sym)
mask    = (sp_mat != 0) & (~np.eye(N_G, dtype=bool))
SIGMA   = 0.15
LAM     = 1e-4

n_pairs = int(mask.sum() // 2)
print(f'Constrained pairs: {n_pairs}  use_prior={not args.no_prior}', flush=True)

# ── gLV helpers ───────────────────────────────────────────────────────────────
def rhs(t, phi, b, A):
    f = b + A @ phi
    return phi * (f - phi @ f)

def integrate_step(phi0, b, A, dt=1.0):
    sol = solve_ivp(rhs, [0, dt], phi0, args=(b, A),
                    method='RK45', rtol=1e-6, atol=1e-8, dense_output=False)
    phi1 = np.clip(sol.y[:, -1], 0, None)
    s = phi1.sum()
    return phi1 / s if s > 1e-12 else phi0.copy()

def sign_penalty(A):
    pen = 0.0
    for i in range(N_G):
        for j in range(N_G):
            if mask[i, j]:
                w = abs(net_sym[i, j])
                v = max(0.0, -sp_mat[i, j] * A[i, j])
                pen += w * v * v / (2 * SIGMA**2)
    return pen

# ── Full-data training objective ──────────────────────────────────────────────
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
    pen  = 0.0 if args.no_prior else sign_penalty(A)
    return rmse + pen + LAM * np.sum(A**2)

# Bounds: diagonal A_ii <= 0
bounds = [(None, 0.0) if i == j else (None, None)
          for i in range(N_G) for j in range(N_G)]
bounds += [(None, None)] * (N_P * N_G)

# ── Warm start ────────────────────────────────────────────────────────────────
rng = np.random.default_rng(0)
A0  = rng.normal(0, 0.01, (N_G, N_G))
np.fill_diagonal(A0, -0.1)
b0  = np.zeros((N_P, N_G))
x0  = np.concatenate([A0.ravel(), b0.ravel()])

print(f'Initial loss = {train_obj(x0):.5f}', flush=True)
t0 = time.time()
cnt = [0]

def fg(x):
    cnt[0] += 1
    v = train_obj(x)
    if cnt[0] % 100 == 1:
        print(f'  iter {cnt[0]:4d}  loss={v:.5f}  t={time.time()-t0:.0f}s', flush=True)
    return v

res = minimize(fg, x0, method='L-BFGS-B', bounds=bounds,
               options={'maxiter': 3000, 'ftol': 1e-10, 'gtol': 1e-7})
print(f'\n{res.message}  iters={cnt[0]}  loss={res.fun:.5f}', flush=True)

A_botelho = res.x[:N_G * N_G].reshape(N_G, N_G)
b_botelho = res.x[N_G * N_G:].reshape(N_P, N_G)

# ── Load Dieckow A ────────────────────────────────────────────────────────────
d_dk = json.load(open(CR / 'fit_glv_hamilton_kegg_expanded_agora_w1p0.json'))
A_dk = np.array(d_dk['A'])

# ── Sign agreement: Botelho A vs Dieckow A ────────────────────────────────────
print('\n── Sign agreement: Botelho A vs Dieckow A ──────────────────────')
pairs_all = [(i, j) for i in range(N_G) for j in range(i+1, N_G)]
pairs_constrained = [(i, j) for i, j in pairs_all if mask[i, j]]

def sign_agree(pairs, A1, A2, label=''):
    matches = [(np.sign(A1[i,j]) == np.sign(A2[i,j])) for i,j in pairs]
    n = len(matches); m = sum(matches)
    p_binom = stats.binom_test(m, n, 0.5, alternative='greater') if n > 0 else 1.0
    print(f'  {label}: {m}/{n} = {m/n:.1%}  p={p_binom:.4f} (binomial vs 50%)')
    return m, n, matches

m_all,  n_all,  _ = sign_agree(pairs_all,         A_botelho, A_dk, 'All 45 pairs')
m_con,  n_con,  _ = sign_agree(pairs_constrained, A_botelho, A_dk, f'Constrained ({n_pairs})')
# Strong pairs only (|A| > 0.1 in BOTH matrices)
strong = [(i,j) for i,j in pairs_all
          if abs(A_botelho[i,j]) > 0.10 and abs(A_dk[i,j]) > 0.10]
if strong:
    m_str, n_str, _ = sign_agree(strong, A_botelho, A_dk,
                                  f'Strong in both (|A|>0.1, n={len(strong)})')

# ── Per-pair table ────────────────────────────────────────────────────────────
print('\n── Top 10 magnitude pairs in Dieckow A (vs Botelho A) ──────────')
dk_pairs_sorted = sorted(pairs_all, key=lambda p: abs(A_dk[p[0], p[1]]), reverse=True)
for i, j in dk_pairs_sorted[:10]:
    agree = '✓' if np.sign(A_dk[i,j]) == np.sign(A_botelho[i,j]) else '✗'
    c_str = '●' if mask[i, j] else ' '
    print(f'  {c_str} {GUILD_ORDER[i][:12]:12} ↔ {GUILD_ORDER[j][:12]:12}  '
          f'Dieckow={A_dk[i,j]:+.3f}  Botelho={A_botelho[i,j]:+.3f}  {agree}')

# ── Training RMSE of Botelho fit ──────────────────────────────────────────────
sq, cnt2 = 0.0, 0
for p in range(N_P):
    phi_pred = phi_all[p, 0].copy()
    for t in range(1, N_T):
        phi_pred = integrate_step(phi_pred, b_botelho[p], A_botelho)
        sq += np.sum((phi_pred - phi_all[p, t])**2)
        cnt2 += N_G
rmse_botelho = np.sqrt(sq / cnt2)
print(f'\nBotelho fit RMSE (in-sample): {rmse_botelho:.4f}', flush=True)

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    'A_botelho': A_botelho.tolist(),
    'A_dieckow': A_dk.tolist(),
    'sign_agreement': {
        'all_pairs':   {'n': n_all, 'agree': m_all, 'pct': round(m_all/n_all*100, 1)},
        'constrained': {'n': n_con, 'agree': m_con, 'pct': round(m_con/n_con*100, 1)},
    },
    'botelho_rmse': float(rmse_botelho),
    'guild_order': GUILD_ORDER,
}
if strong:
    result['sign_agreement']['strong'] = {
        'n': n_str, 'agree': m_str, 'pct': round(m_str/n_str*100, 1)}

json.dump(result, open(OUT / 'botelho_A_comparison.json', 'w'), indent=2)
print(f'Saved → {OUT}/botelho_A_comparison.json', flush=True)

# ── Figure ────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 10,
})

ABBR = ['Actin', 'Bacil', 'Bctrd', 'β-Pro', 'Clost',
        'Corio', 'Fusob', 'γ-Pro', 'Negat', 'Other']

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, A, title in zip(axes,
    [A_dk, A_botelho, None],
    ['(A) Dieckow A\n(AGORA W=1.0, 10 patients, 3 weeks)',
     '(B) Botelho A\n(15 patients, 7 months)',
     '(C) Sign agreement heatmap']):

    if A is not None:
        vmax = max(abs(A[~np.eye(N_G, dtype=bool)]).max(), 0.5)
        im = ax.imshow(A, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8, label=r'$A_{ij}$')
        ax.set_xticks(range(N_G)); ax.set_yticks(range(N_G))
        ax.set_xticklabels(ABBR, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ABBR, fontsize=8)
        ax.set_title(title, fontsize=9)
    else:
        # Sign agreement heatmap
        agree_mat = np.zeros((N_G, N_G))
        for i in range(N_G):
            for j in range(N_G):
                if i != j:
                    agree_mat[i, j] = 1 if np.sign(A_dk[i,j]) == np.sign(A_botelho[i,j]) else -1
        agree_mat[np.eye(N_G, dtype=bool)] = np.nan
        im3 = ax.imshow(agree_mat, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im3, ax=ax, shrink=0.8, label='Sign: ✓(+1) / ✗(-1)')
        ax.set_xticks(range(N_G)); ax.set_yticks(range(N_G))
        ax.set_xticklabels(ABBR, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ABBR, fontsize=8)
        for i, j in pairs_constrained:
            for ri, rj in [(i,j),(j,i)]:
                ax.add_patch(plt.Rectangle((rj-0.5, ri-0.5), 1, 1,
                             fill=False, edgecolor='black', lw=1.4))
        ax.set_title(f'(C) Sign agreement: Dieckow vs Botelho\n'
                     f'All: {m_all}/{n_all}={m_all/n_all:.0%}  '
                     f'Constrained: {m_con}/{n_con}={m_con/n_con:.0%}',
                     fontsize=9)

fig.suptitle('Cross-dataset A matrix comparison: Dieckow 2024 vs Botelho 2021',
             fontsize=11, y=1.02)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(OUT / f'fig_botelho_A_comparison.{ext}', dpi=150, bbox_inches='tight')
    print(f'Saved {OUT}/fig_botelho_A_comparison.{ext}')
