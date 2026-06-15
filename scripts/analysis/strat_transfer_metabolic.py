#!/usr/bin/env python3
"""
strat_transfer_metabolic.py — Does the cross-cohort sign agreement come from the
*metabolically grounded* (cross-feeding) interactions?

Motivation (Szafranski 2026-06-14 + reviews): Dieckow (early peri-implant
colonization, weeks) and Duran-Pinedo 2021 (periodontitis progression, months)
differ in design, so a blanket "cross-cohort agreement" claim is too strong.
The design-INVARIANT part should be metabolism (cross-feeding does not care about
disease state), while competition is context-dependent. So we stratify the
prior-free Dieckow x Duran-Pinedo sign agreement and test whether facilitation
(cross-feeding) interactions transfer across cohorts better than competition.

Two stratification axes:
  (1) PRIMARY  — by interaction TYPE (sign in Dieckow): facilitation (+) vs
      competition (-). Prior-free; directly mirrors the "cross-feeding validated,
      competition not" finding on the cross-cohort axis.
  (2) SECONDARY — by metabolic membership M = the 36 unordered pairs among the 9
      AGORA-modelled guilds (the AGORA cross-feeding prior is all-positive) vs the
      non-metabolic pairs involving the residual 'Other' guild. NOTE: confounded
      because 'Other' is a heterogeneous catch-all; reported with that caveat.

Honest about power: n is small (45 unordered pairs; ~8-9 strong). Results are
DESCRIPTIVE / consistency checks, not powered tests. Fisher exact p reported but
read as effect-direction, not significance.

Run (from repo root):  python scripts/analysis/strat_transfer_metabolic.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports
import json
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from guild_replicator_dieckow import GUILD_ORDER, N_G

ROOT = _pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "duranpinedo_validation"

A_dk = np.array(json.load(open(OUT / "dieckow_A_noprior_comparison.json"))["A_dieckow_noprior"])
A_bot = np.array(json.load(open(OUT / "duranpinedo_A_comparison_noprior.json"))["A_duranpinedo"])

# ── metabolic set M (unordered pairs with a non-zero AGORA cross-feeding sign) ──
idx = {g: k for k, g in enumerate(GUILD_ORDER)}
agora = json.load(open(ROOT / "results" / "dieckow_cr" / "agora_sign_comparison.json"))
M_pairs = set()
for e in agora:
    if e.get("sign_agora", 0) != 0:
        i, j = idx[e["src"]], idx[e["tgt"]]
        M_pairs.add((min(i, j), max(i, j)))

pairs = [(i, j) for i in range(N_G) for j in range(i + 1, N_G)]  # 45 unordered

def sym(A):
    return (A + A.T) / 2.0

def vals(A, mode):
    A = sym(A) if mode == "sym" else A
    return np.array([A[i, j] for i, j in pairs])

def frac(agree_mask):
    n = int(agree_mask.size)
    a = int(agree_mask.sum())
    return a, n, (a / n if n else float("nan"))

def wconc(a, b):
    w = np.abs(a) * np.abs(b)
    return float((w * np.sign(a) * np.sign(b)).sum() / w.sum()) if w.sum() else float("nan")

def block(mode, theta):
    a = vals(A_dk, mode)
    b = vals(A_bot, mode)
    inM = np.array([(i, j) in M_pairs for (i, j) in pairs])
    strong = (np.abs(a) > theta) & (np.abs(b) > theta)
    agree = np.sign(a) == np.sign(b)

    res = {"mode": mode, "theta": theta}

    # (1) by interaction TYPE (Dieckow sign), among strong pairs
    fac = strong & (a > 0)
    comp = strong & (a < 0)
    af, nf, ff = frac(agree[fac])
    ac, nc, fc = frac(agree[comp])
    res["by_type"] = {
        "facilitation": {"agree": af, "n": nf, "frac": ff},
        "competition": {"agree": ac, "n": nc, "frac": fc},
    }
    # Fisher exact: type x (agree/disagree)
    table = [[af, nf - af], [ac, nc - ac]]
    try:
        res["by_type"]["fisher_p"] = float(stats.fisher_exact(table, alternative="greater")[1])
    except Exception:
        res["by_type"]["fisher_p"] = None

    # (2) by metabolic membership, among strong pairs
    sM = strong & inM
    sN = strong & ~inM
    aM, nM, fM = frac(agree[sM])
    aN, nN, fN = frac(agree[sN])
    res["by_metabolic"] = {
        "in_M": {"agree": aM, "n": nM, "frac": fM},
        "not_M": {"agree": aN, "n": nN, "frac": fN},
        "fisher_p": float(stats.fisher_exact([[aM, nM - aM], [aN, nN - aN]],
                                             alternative="greater")[1])
        if (nM and nN) else None,
    }

    # threshold-free weighted concordance per group
    res["wconc"] = {
        "facilitation": wconc(a[a > 0], b[a > 0]),
        "competition": wconc(a[a < 0], b[a < 0]),
        "in_M": wconc(a[inM], b[inM]),
        "not_M": wconc(a[~inM], b[~inM]),
    }
    # directional: of strong metabolic pairs, fraction where BOTH cohorts positive
    bothpos = sM & (a > 0) & (b > 0)
    res["metabolic_both_facilitation"] = {"k": int(bothpos.sum()), "n_strong_M": int(sM.sum())}
    return res

THETA = 0.10
results = {"theta": THETA, "n_pairs": len(pairs), "n_M_pairs": len(M_pairs),
           "note": "Descriptive/consistency (small n); Fisher p = effect direction, not powered."}
for mode in ("raw", "sym"):
    results[mode] = block(mode, THETA)

# ── console summary ───────────────────────────────────────────────────────────
for mode in ("raw", "sym"):
    r = results[mode]
    bt, bm = r["by_type"], r["by_metabolic"]
    print(f"\n===== MODE={mode.upper()}  (strong pairs |A|>{THETA}) =====")
    print(f"  BY TYPE (Dieckow sign):")
    print(f"    facilitation (+): {bt['facilitation']['agree']}/{bt['facilitation']['n']}"
          f" = {bt['facilitation']['frac']:.0%}   (W={r['wconc']['facilitation']:+.2f})")
    print(f"    competition  (-): {bt['competition']['agree']}/{bt['competition']['n']}"
          f" = {bt['competition']['frac']:.0%}   (W={r['wconc']['competition']:+.2f})")
    print(f"    Fisher p (fac>comp) = {bt['fisher_p']}")
    print(f"  BY METABOLIC SET:")
    print(f"    in M  (AGORA cross-feeding): {bm['in_M']['agree']}/{bm['in_M']['n']}"
          f" = {bm['in_M']['frac']:.0%}   (W={r['wconc']['in_M']:+.2f})")
    print(f"    not M (involves 'Other'):    {bm['not_M']['agree']}/{bm['not_M']['n']}"
          f" = {bm['not_M']['frac'] if bm['not_M']['n'] else float('nan')}   (W={r['wconc']['not_M']:+.2f})")
    md = r["metabolic_both_facilitation"]
    print(f"    strong M pairs positive in BOTH cohorts: {md['k']}/{md['n_strong_M']}")

json.dump(results, open(OUT / "strat_transfer_metabolic.json", "w"), indent=2)
print(f"\nSaved -> {OUT}/strat_transfer_metabolic.json")

# ── figure (raw mode headline) ────────────────────────────────────────────────
r = results["raw"]
plt.rcParams.update({"font.size": 9, "font.family": "serif"})
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))

g = ax[0]
bt = r["by_type"]
labels = [f"facilitation\n(+, n={bt['facilitation']['n']})", f"competition\n(−, n={bt['competition']['n']})"]
fr = [bt["facilitation"]["frac"], bt["competition"]["frac"]]
bars = g.bar(labels, [0 if np.isnan(x) else x for x in fr], color=["#2a7", "#c44"], width=0.6)
g.axhline(0.5, ls="--", c="gray", lw=0.8)
g.set_ylim(0, 1.05); g.set_ylabel("cross-cohort sign agreement")
g.set_title("(A) by interaction type", fontsize=9)
for bar, x, n in zip(bars, fr, [bt["facilitation"]["n"], bt["competition"]["n"]]):
    g.text(bar.get_x() + bar.get_width() / 2, (0 if np.isnan(x) else x) + 0.02,
           "n/a" if np.isnan(x) else f"{x:.0%}", ha="center", fontsize=8)

g = ax[1]
bm = r["by_metabolic"]
labels = [f"metabolic M\n(n={bm['in_M']['n']})", f"non-metabolic\n(n={bm['not_M']['n']})"]
fr = [bm["in_M"]["frac"], bm["not_M"]["frac"]]
bars = g.bar(labels, [0 if (x is None or np.isnan(x)) else x for x in fr], color=["#27a", "#caa"], width=0.6)
g.axhline(0.5, ls="--", c="gray", lw=0.8)
g.set_ylim(0, 1.05)
g.set_title("(B) by metabolic set (caveat: 'Other'=catch-all)", fontsize=8.5)
for bar, x in zip(bars, fr):
    val = 0 if (x is None or np.isnan(x)) else x
    g.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
           "n/a" if (x is None or np.isnan(x)) else f"{x:.0%}", ha="center", fontsize=8)

fig.suptitle("Cross-cohort transfer (Dieckow × Duran-Pinedo, prior-free): "
             "stratification underpowered — both types largely concordant (n too small)", fontsize=8.8)
fig.tight_layout(rect=[0, 0, 1, 0.93])
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"fig_strat_transfer_metabolic.{ext}", dpi=200, bbox_inches="tight")
print(f"Saved -> {OUT}/fig_strat_transfer_metabolic.(pdf|png)")
