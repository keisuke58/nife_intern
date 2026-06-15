#!/usr/bin/env python3
"""
outlook_hierarchical_glv.py — OUTLOOK prototype: partial-pooling cross-cohort gLV.

Models the design gap instead of comparing point fits (the principled version of
"do the cohorts share an interaction structure?"). Each cohort uses
    A_cohort = A_shared + delta_cohort,
with delta penalised toward 0 (lambda_dev). This deterministic ridge prototype
interpolates between fully-separate fits (lambda_dev -> 0) and one forced-shared A
(lambda_dev -> inf, == verify_shared_A V4). It is the cheap stand-in for a full
hierarchical Bayesian gLV (shared A drawn from a hyperprior, per-cohort deviations
with a learned scale) which is the real Outlook and belongs on PBS / PyMC.

Reports, per lambda_dev: total & per-cohort RMSE, ||delta|| per cohort (how much
deviation each cohort needs), and the shared backbone's strong-pair signs.

Bounded compute (shared node): fast fixed-step RK4, vectorized, capped iters.

Run (repo root):  python scripts/analysis/outlook_hierarchical_glv.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
import json, time
import numpy as np
from scipy.optimize import minimize

from guild_replicator_dieckow import N_G, GUILD_ORDER

ROOT = _pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "duranpinedo_validation"

DK = np.load(ROOT / "results" / "dieckow_otu" / "phi_guild.npy")   # (10,3,10)
BOT = np.load(ROOT / "data" / "prjna725874" / "phi_guild.npy")     # (15,7,10)
G = N_G
P_DK, P_BOT = DK.shape[0], BOT.shape[0]
LAM_A = 1e-4
N_SUB = 8
MAXITER = 150

def rhs(phi, b, A):
    f = b + phi @ A.T
    return phi * (f - (phi * f).sum(1, keepdims=True))

def step(phi, b, A):
    h = 1.0 / N_SUB
    for _ in range(N_SUB):
        k1 = rhs(phi, b, A); k2 = rhs(phi + 0.5 * h * k1, b, A)
        k3 = rhs(phi + 0.5 * h * k2, b, A); k4 = rhs(phi + h * k3, b, A)
        phi = phi + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    phi = np.clip(phi, 0, None); s = phi.sum(1, keepdims=True)
    return np.where(s > 1e-12, phi / s, phi)

def rollout_sq(X, b, A):
    """X (n,T,G), b (n,G), A (G,G) -> (sum sq err, count)."""
    pr = X[:, 0, :].copy(); sq = 0
    for t in range(1, X.shape[1]):
        pr = step(pr, b, A); sq += np.sum((pr - X[:, t, :]) ** 2)
    return sq, (X.shape[1] - 1) * X.shape[0] * G

NA = G * G
def unpack(x):
    As = x[:NA].reshape(G, G)
    ddk = x[NA:2 * NA].reshape(G, G)
    ddp = x[2 * NA:3 * NA].reshape(G, G)
    b = x[3 * NA:].reshape(P_DK + P_BOT, G)
    return As, ddk, ddp, b

def make_obj(lam_dev):
    def obj(x):
        As, ddk, ddp, b = unpack(x)
        A_dk, A_dp = As + ddk, As + ddp
        sq1, c1 = rollout_sq(DK, b[:P_DK], A_dk)
        sq2, c2 = rollout_sq(BOT, b[P_DK:], A_dp)
        rmse = np.sqrt((sq1 + sq2) / (c1 + c2))
        return (rmse + lam_dev * (np.sum(ddk ** 2) + np.sum(ddp ** 2))
                + LAM_A * np.sum(As ** 2))
    return obj

def fit(lam_dev, seed=0):
    rng = np.random.default_rng(seed)
    As0 = rng.normal(0, 0.01, (G, G)); np.fill_diagonal(As0, -0.1)
    x0 = np.concatenate([As0.ravel(), np.zeros(NA), np.zeros(NA),
                         np.zeros((P_DK + P_BOT) * G)])
    # bound shared-A diagonal <= 0; deltas & b free
    bounds = [(None, 0.0) if i == j else (None, None) for i in range(G) for j in range(G)]
    bounds += [(None, None)] * (2 * NA + (P_DK + P_BOT) * G)
    r = minimize(make_obj(lam_dev), x0, method="L-BFGS-B", bounds=bounds,
                 options={"maxiter": MAXITER, "maxfun": 60000, "ftol": 1e-9, "gtol": 1e-6})
    return r.x

pairs = [(i, j) for i in range(G) for j in range(i + 1, G)]
def upper(A): return np.array([A[i, j] for i, j in pairs])
def strong_agree(a, b, th=0.10):
    sel = (np.abs(a) > th) & (np.abs(b) > th); n = int(sel.sum())
    return (int((np.sign(a[sel]) == np.sign(b[sel])).sum()), n) if n else (0, 0)

t0 = time.time()
LAMS = [0.0, 0.003, 0.03, 1.0]   # 0 ~ separate ; large ~ forced shared
rows = []
for lam in LAMS:
    x = fit(lam)
    As, ddk, ddp, b = unpack(x)
    A_dk, A_dp = As + ddk, As + ddp
    sq1, c1 = rollout_sq(DK, b[:P_DK], A_dk)
    sq2, c2 = rollout_sq(BOT, b[P_DK:], A_dp)
    rmse_dk, rmse_dp = np.sqrt(sq1 / c1), np.sqrt(sq2 / c2)
    ag, n = strong_agree(upper(A_dk), upper(A_dp))
    row = {"lambda_dev": lam, "rmse_dk": float(rmse_dk), "rmse_dp": float(rmse_dp),
           "norm_delta_dk": float(np.linalg.norm(ddk)), "norm_delta_dp": float(np.linalg.norm(ddp)),
           "shared_strong_pairs": int(((np.abs(upper(As)) > 0.10)).sum()),
           "Adk_Adp_strong_agree": [ag, n]}
    rows.append(row)
    print(f"lam={lam:<6}  RMSE dk={rmse_dk:.4f} dp={rmse_dp:.4f}  "
          f"||d_dk||={row['norm_delta_dk']:.2f} ||d_dp||={row['norm_delta_dp']:.2f}  "
          f"A_dk~A_dp {ag}/{n}  t={time.time()-t0:.0f}s", flush=True)

# shared backbone (strong-pair signs) at the strongest pooling that still fits (lam=0.03)
x = fit(0.03); As, ddk, ddp, b = unpack(x)
backbone = []
for (i, j) in pairs:
    if abs(As[i, j]) > 0.10 or abs(As[j, i]) > 0.10:
        backbone.append({"pair": [GUILD_ORDER[i], GUILD_ORDER[j]],
                         "A_shared_ij": float(As[i, j]), "A_shared_ji": float(As[j, i])})

res = {"settings": {"maxiter": MAXITER, "rk4_substeps": N_SUB, "lambdas": LAMS},
       "sweep": rows, "shared_backbone_lam0.03": backbone,
       "note": "Deterministic partial-pooling prototype for the hierarchical-Bayesian "
               "Outlook. lambda_dev->0 ~ separate fits; large ~ forced-shared (V4). "
               "delta norm shows how much each cohort deviates from the shared backbone."}
json.dump(res, open(OUT / "outlook_hierarchical_glv.json", "w"), indent=2)
print(f"\nShared backbone (|A_shared|>0.1) strong pairs: {len(backbone)}")
print(f"Saved -> {OUT}/outlook_hierarchical_glv.json   (total {time.time()-t0:.0f}s)")
