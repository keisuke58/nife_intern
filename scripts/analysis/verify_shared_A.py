#!/usr/bin/env python3
"""
verify_shared_A.py — V3 (Duran-Pinedo multi-seed) + V4 (shared-A pooled fit).

V3: refit the Duran-Pinedo prior-free gLV with several seeds -> per-interaction
    sign stability (the DP analogue of the Dieckow seed check). Turns the
    cross-cohort comparison from point-vs-point into ensemble-vs-ensemble.

V4 (the proper cross-cohort test): fit ONE SHARED interaction matrix A on the
    POOLED patients of both cohorts (each patient keeps its own b and its own
    timepoints), and compare its per-cohort RMSE to the per-cohort SEPARATE fits.
    If a single A explains both cohorts almost as well as cohort-specific A's,
    that is direct evidence of a common interaction backbone -- the principled
    version of "the cohorts agree", without relying on a 9-pair sign count.

Bounded compute (shared login node): looser integration + capped iters, few seeds.
Writes only NEW files; never overwrites the canonical fits.

Run (repo root):  python scripts/analysis/verify_shared_A.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
import json, time
import numpy as np
from scipy.optimize import minimize

from guild_replicator_dieckow import N_G

ROOT = _pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "duranpinedo_validation"

DK = np.load(ROOT / "results" / "dieckow_otu" / "phi_guild.npy")   # (10,3,10)
BOT = np.load(ROOT / "data" / "prjna725874" / "phi_guild.npy")     # (15,7,10)

LAM = 1e-4
MAXITER = 150      # loss converges by ~100 iters (benchmarked); bounded for shared node
N_SUB = 8          # fixed RK4 substeps over dt=1 (ample accuracy for smooth dt=1 dynamics)

def rhs_vec(phi, b, A):
    """Vectorized over patients. phi,b: (n,G); A: (G,G). returns (n,G)."""
    f = b + phi @ A.T
    proj = (phi * f).sum(axis=1, keepdims=True)
    return phi * (f - proj)

def step_vec(phi, b, A):
    """One unit-time replicator step, RK4, vectorized over patients (n,G)."""
    h = 1.0 / N_SUB
    for _ in range(N_SUB):
        k1 = rhs_vec(phi, b, A)
        k2 = rhs_vec(phi + 0.5 * h * k1, b, A)
        k3 = rhs_vec(phi + 0.5 * h * k2, b, A)
        k4 = rhs_vec(phi + h * k3, b, A)
        phi = phi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    phi = np.clip(phi, 0, None)
    s = phi.sum(axis=1, keepdims=True)
    return np.where(s > 1e-12, phi / s, phi)

def _groups(phis):
    """group patient indices by trajectory length T (for vectorized rollout)."""
    g = {}
    for idx, xp in enumerate(phis):
        g.setdefault(xp.shape[0], []).append(idx)
    return {T: (np.array(ix), np.stack([phis[i] for i in ix])) for T, ix in g.items()}

def make_obj(phis):
    """phis: list of (T_p, G). Shared A, per-patient b. Vectorized within T-groups."""
    P = len(phis)
    grp = _groups(phis)              # T -> (idx array, stacked (n,T,G))

    def predict_sq(A, b):
        sq = cnt = 0
        for T, (idx, X) in grp.items():
            pr = X[:, 0, :].copy()
            bg = b[idx]
            for t in range(1, T):
                pr = step_vec(pr, bg, A)
                sq += np.sum((pr - X[:, t, :]) ** 2); cnt += idx.size * N_G
        return sq, cnt

    def obj(x):
        A = x[:N_G * N_G].reshape(N_G, N_G)
        b = x[N_G * N_G:].reshape(P, N_G)
        sq, cnt = predict_sq(A, b)
        return np.sqrt(sq / cnt) + LAM * np.sum(A ** 2)
    return obj, P

def fit(phis, seed):
    obj, P = make_obj(phis)
    rng = np.random.default_rng(seed)
    A0 = rng.normal(0, 0.01, (N_G, N_G)); np.fill_diagonal(A0, -0.1)
    x0 = np.concatenate([A0.ravel(), np.zeros(P * N_G)])
    bounds = [(None, 0.0) if i == j else (None, None) for i in range(N_G) for j in range(N_G)]
    bounds += [(None, None)] * (P * N_G)
    r = minimize(obj, x0, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": MAXITER, "maxfun": 30000, "ftol": 1e-9, "gtol": 1e-6})
    A = r.x[:N_G * N_G].reshape(N_G, N_G)
    b = r.x[N_G * N_G:].reshape(P, N_G)
    return A, b, float(r.fun)

def rmse_of(A, b, phis):
    grp = _groups(phis)
    sq = cnt = 0
    for T, (idx, X) in grp.items():
        pr = X[:, 0, :].copy(); bg = b[idx]
        for t in range(1, T):
            pr = step_vec(pr, bg, A); sq += np.sum((pr - X[:, t, :]) ** 2); cnt += idx.size * N_G
    return float(np.sqrt(sq / cnt))

dk_list = [DK[p] for p in range(DK.shape[0])]
bot_list = [BOT[p] for p in range(BOT.shape[0])]
pairs = [(i, j) for i in range(N_G) for j in range(i + 1, N_G)]
def upper(A): return np.array([A[i, j] for i, j in pairs])

t0 = time.time()
print("[V3] Duran-Pinedo multi-seed fits ...", flush=True)
SEEDS = [0, 1, 2]
dp_As = []
dp_fits = {}
for s in SEEDS:
    A, b, loss = fit(bot_list, s)
    dp_As.append(A)
    dp_fits[s] = (A, b)
    print(f"   seed {s}: loss={loss:.5f}  t={time.time()-t0:.0f}s", flush=True)

uv = [upper(A) for A in dp_As]
signs = np.sign(np.stack(uv)); maj = np.sign(signs.sum(0))
consist = (signs == maj).mean(0)
strong_any = (np.abs(np.stack(uv)) > 0.10).any(0)
dp_ident = float(consist[strong_any].mean()) if strong_any.any() else None
print(f"[V3] DP per-interaction sign consistency (strong pairs): {dp_ident:.1%}", flush=True)

print("\n[V4] separate vs shared-A pooled fits ...", flush=True)
A_dk, b_dk, _ = fit(dk_list, 0)
rmse_dk_sep = rmse_of(A_dk, b_dk, dk_list)
A_bot, b_bot = dp_fits[0]          # reuse DP seed-0 fit (== separate DP fit)
rmse_bot_sep = rmse_of(A_bot, b_bot, bot_list)
print(f"   separate: Dieckow RMSE={rmse_dk_sep:.4f}  DP RMSE={rmse_bot_sep:.4f}  t={time.time()-t0:.0f}s", flush=True)

A_sh, b_sh, _ = fit(dk_list + bot_list, 0)
b_sh_dk, b_sh_bot = b_sh[:len(dk_list)], b_sh[len(dk_list):]
rmse_dk_sh = rmse_of(A_sh, b_sh_dk, dk_list)
rmse_bot_sh = rmse_of(A_sh, b_sh_bot, bot_list)
print(f"   shared-A: Dieckow RMSE={rmse_dk_sh:.4f}  DP RMSE={rmse_bot_sh:.4f}  t={time.time()-t0:.0f}s", flush=True)

# how much worse is one shared A vs cohort-specific A's? (relative)
pct_dk = 100 * (rmse_dk_sh - rmse_dk_sep) / rmse_dk_sep
pct_bot = 100 * (rmse_bot_sh - rmse_bot_sep) / rmse_bot_sep
print(f"\n[V4] shared-A penalty: Dieckow +{pct_dk:.1f}%   DP +{pct_bot:.1f}%  RMSE vs separate")
print("     (small penalty => a single interaction matrix explains BOTH cohorts)")

res = {
    "settings": {"maxiter": MAXITER, "rk4_substeps": N_SUB, "dp_seeds": SEEDS},
    "v3_dp_sign_identifiability_strong": dp_ident,
    "v4": {
        "rmse_dieckow_separate": rmse_dk_sep, "rmse_dieckow_shared": rmse_dk_sh,
        "rmse_dp_separate": rmse_bot_sep, "rmse_dp_shared": rmse_bot_sh,
        "shared_penalty_pct_dieckow": pct_dk, "shared_penalty_pct_dp": pct_bot,
    },
    "note": "Fixed-step RK4 integration (speed); internally consistent across all "
            "fits here. Compare shared-vs-separate RMSE for the common-backbone claim.",
}
json.dump(res, open(OUT / "verify_shared_A.json", "w"), indent=2)
print(f"\nSaved -> {OUT}/verify_shared_A.json   (total {time.time()-t0:.0f}s)")
