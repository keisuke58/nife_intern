#!/usr/bin/env python3
"""
verify_seed_identifiability.py — Is the cross-cohort 89% above or below the
WITHIN-cohort fit-reproducibility ceiling?

The prior-free gLV fit is non-convex, so two random-seed fits of the SAME cohort
already disagree somewhat on signs. That within-cohort cross-seed agreement is the
right yardstick for the cross-cohort number: if Dieckow's own seeds only agree X%,
then "Dieckow x Duran-Pinedo = 89%" must be read against X, not against 100%.

Uses the 5 existing Dieckow no-prior seed fits (no refitting):
  results/duranpinedo_validation/A_dk_noprior_seed{1..5}.npy

Outputs:
  - within-Dieckow cross-seed strong-pair sign agreement (all C(5,2)=10 pairs);
  - per-interaction sign identifiability = fraction of seeds sharing the majority
    sign (a practical-identifiability proxy, cf. structural identifiability of gLV,
    Remien 2021 / van den Berg 2022);
  - the cross-cohort 8/9 in context.

Run (repo root):  python scripts/analysis/verify_seed_identifiability.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
import json
from itertools import combinations
import numpy as np

from guild_replicator_dieckow import N_G

ROOT = _pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "duranpinedo_validation"

seeds = [np.load(OUT / f"A_dk_noprior_seed{k}.npy") for k in range(1, 6)]
A_bot = np.array(json.load(open(OUT / "duranpinedo_A_comparison_noprior.json"))["A_duranpinedo"])

pairs = [(i, j) for i in range(N_G) for j in range(i + 1, N_G)]
THETA = 0.10

def upper(A):
    return np.array([A[i, j] for i, j in pairs])

def strong_agree(a, b, theta=THETA):
    sel = (np.abs(a) > theta) & (np.abs(b) > theta)
    n = int(sel.sum())
    if n == 0:
        return float("nan"), 0, 0
    ag = int((np.sign(a[sel]) == np.sign(b[sel])).sum())
    return ag / n, ag, n

# ── within-Dieckow cross-seed agreement (all seed pairs) ──────────────────────
uv = [upper(s) for s in seeds]
within = []
for i, j in combinations(range(len(seeds)), 2):
    frac, ag, n = strong_agree(uv[i], uv[j])
    within.append({"seeds": [i + 1, j + 1], "frac": frac, "agree": ag, "n": n})
fr = np.array([w["frac"] for w in within if w["n"] > 0])

# ── per-interaction sign identifiability across the 5 seeds ───────────────────
stk = np.stack(uv)                      # (5, 45)
signs = np.sign(stk)
maj = np.sign(signs.sum(axis=0))
consist = (signs == maj).mean(axis=0)   # fraction of seeds sharing majority sign
strong_any = (np.abs(stk) > THETA).any(axis=0)
ident_strong = consist[strong_any]

# ── cross-cohort, using the seed-consensus Dieckow vs Duran-Pinedo ────────────
A_dk_consensus = np.median(np.stack(seeds), axis=0)
frac_x, ag_x, n_x = strong_agree(upper(A_dk_consensus), upper(A_bot))

res = {
    "theta": THETA,
    "within_dieckow_crossseed": {
        "mean_frac": float(np.mean(fr)), "min_frac": float(np.min(fr)),
        "max_frac": float(np.max(fr)), "n_seed_pairs": len(fr),
        "detail": within,
    },
    "per_interaction_sign_identifiability": {
        "mean_all": float(consist.mean()),
        "mean_strong_pairs": float(ident_strong.mean()) if ident_strong.size else None,
        "n_strong_pairs": int(strong_any.sum()),
        "fully_consistent_strong": int((ident_strong == 1.0).sum()) if ident_strong.size else 0,
    },
    "cross_cohort_consensus": {"frac": frac_x, "agree": ag_x, "n": n_x},
}

print("=== Within-Dieckow cross-seed strong-pair sign agreement (the ceiling) ===")
print(f"  mean {np.mean(fr):.1%}   range [{np.min(fr):.0%}, {np.max(fr):.0%}]   ({len(fr)} seed-pairs)")
print("\n=== Per-interaction sign identifiability across 5 seeds ===")
print(f"  all pairs: mean {consist.mean():.1%} of seeds share the sign")
print(f"  strong pairs (|A|>{THETA} in any seed, n={int(strong_any.sum())}): "
      f"mean {ident_strong.mean():.1%}; fully-consistent {int((ident_strong==1.0).sum())}/{ident_strong.size}")
print("\n=== Cross-cohort (Dieckow seed-consensus × Duran-Pinedo) ===")
print(f"  {ag_x}/{n_x} = {frac_x:.1%}")
print("\nRead: compare the cross-cohort % to the within-Dieckow ceiling above. If they")
print("are similar, cross-cohort agreement is as good as re-fitting the SAME cohort.")

json.dump(res, open(OUT / "verify_seed_identifiability.json", "w"), indent=2)
print(f"\nSaved -> {OUT}/verify_seed_identifiability.json")
