#!/usr/bin/env python3
"""
verify_design_difference.py — Quantify HOW the two cohorts differ (design gap).

Szafranski's point: Dieckow (early peri-implant colonization, ~weekly, 3 tp) and
Duran-Pinedo 2021 (periodontitis progression, ~2-monthly, 7 tp) are different
designs, so cross-cohort claims must be limited. This makes that gap numerical
instead of rhetorical, using model-free descriptors of the guild trajectories
(à la Duran-Pinedo's own ecological framing: turnover, directionality, synchrony).

Metrics per patient (then compared across cohorts):
  - turnover: mean Bray-Curtis between CONSECUTIVE timepoints.
  - directionality: slope of BC(t0, t) vs t (>0 divergent/directional from baseline,
    <=0 stable/convergent).
  - net-displacement ratio = BC(t0, t_last) / sum(consecutive BC) in [0,1]:
    ~1 = purely directional (succession), ~0 = returns to start (fluctuating).
    (Loreau-de Mazancourt synchrony is ill-defined here: on the simplex the
    community total is fixed at 1, so variance-of-total is ~0 — dropped.)

Honest caveat: Dieckow has only 3 timepoints, so its slopes are coarse.

Run (repo root):  python scripts/analysis/verify_design_difference.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
import json
import numpy as np
from scipy import stats

ROOT = _pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "duranpinedo_validation"

DK = np.load(ROOT / "results" / "dieckow_otu" / "phi_guild.npy")     # (10,3,10)
BOT = np.load(ROOT / "data" / "prjna725874" / "phi_guild.npy")       # (15,7,10)

def bray_curtis(u, v):
    s = np.abs(u - v).sum()
    d = (u + v).sum()
    return s / d if d > 0 else 0.0

def per_patient_metrics(phi):
    """phi: (P, T, G). returns dict of arrays length P."""
    P, T, G = phi.shape
    turnover, slope, net_disp = [], [], []
    for p in range(P):
        x = phi[p]  # (T,G)
        # turnover = mean consecutive BC
        bc_consec = [bray_curtis(x[t], x[t + 1]) for t in range(T - 1)]
        path_len = float(np.sum(bc_consec))
        turnover.append(np.mean(bc_consec))
        # directionality: BC from baseline vs time, slope
        bc0 = np.array([bray_curtis(x[0], x[t]) for t in range(T)])
        sl = np.polyfit(np.arange(T), bc0, 1)[0]
        slope.append(sl)
        # net-displacement ratio: net move from start / total path length
        nd = (bray_curtis(x[0], x[-1]) / path_len) if path_len > 0 else np.nan
        net_disp.append(nd)
    return {"turnover": np.array(turnover), "slope": np.array(slope),
            "net_disp_ratio": np.array(net_disp)}

mdk = per_patient_metrics(DK)
mbot = per_patient_metrics(BOT)

def summ(a):
    a = a[~np.isnan(a)]
    return {"mean": float(np.mean(a)), "sd": float(np.std(a)),
            "median": float(np.median(a)), "n": int(a.size)}

res = {"dieckow": {}, "duranpinedo": {}, "mannwhitney_p": {},
       "shapes": {"dieckow": list(DK.shape), "duranpinedo": list(BOT.shape)},
       "note": "Dieckow has 3 timepoints; slope/synchrony are coarse for it."}
for k in ("turnover", "slope", "net_disp_ratio"):
    res["dieckow"][k] = summ(mdk[k])
    res["duranpinedo"][k] = summ(mbot[k])
    a, b = mdk[k][~np.isnan(mdk[k])], mbot[k][~np.isnan(mbot[k])]
    try:
        res["mannwhitney_p"][k] = float(stats.mannwhitneyu(a, b, alternative="two-sided")[1])
    except Exception:
        res["mannwhitney_p"][k] = None

print("=== Design-difference metrics (per-patient mean ± sd) ===")
print(f"{'metric':14} {'Dieckow (10×3)':>22} {'Duran-Pinedo (15×7)':>24}   MWU p")
for k in ("turnover", "slope", "net_disp_ratio"):
    d, b = res["dieckow"][k], res["duranpinedo"][k]
    print(f"{k:14} {d['mean']:+.3f} ± {d['sd']:.3f}      "
          f"{b['mean']:+.3f} ± {b['sd']:.3f}        {res['mannwhitney_p'][k]:.3f}")
print("\nInterpretation: higher turnover / |slope| => more dynamic & directional change;")
print("net-disp ratio ~1 => directional succession; ~0 => fluctuating around a state.")

json.dump(res, open(OUT / "verify_design_difference.json", "w"), indent=2)
print(f"\nSaved -> {OUT}/verify_design_difference.json")
