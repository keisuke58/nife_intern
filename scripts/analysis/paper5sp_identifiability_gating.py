#!/usr/bin/env python3
"""paper5sp_identifiability_gating.py — reviewer-defense numbers for the
5-species Heine ODE paper, computed from the *canonical* posterior.

Two analyses, both straight from `results/ultimate_10000p` (= paper_data.py)
and the fit-time uniform prior in Tmcmc202601 `prior_bounds.json` — no re-fit:

  (2) Posterior contraction  C_k = 1 - Var_post/Var_prior  for every theta,
      flagging data-informed (C>0.5) vs prior-dominated interaction terms.
  (3) Gating edge: among the four *-Pg interactions, the one that is ~0 in
      commensal states but active + well-contracted under dysbiotic-HOBIC
      (the condition of maximal Pg surge). Identifies A:Fn-Pg.

theta layout / species order follow guild-free attractor_analysis.theta_to_matrices.

Usage (from repo root):  python scripts/analysis/paper5sp_identifiability_gating.py
"""
import sys as _sys, pathlib as _pathlib  # noqa: E402  [nife-pathshim]
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))  # repo root: bare sibling imports

import json
import numpy as np
from paper_data import paper_5sp_samples

# fit-time prior bounds (uniform) live in the TMCMC project, not this repo
PRIOR_JSON = _pathlib.Path(
    '/home/nishioka/IKM_Hiwi/Tmcmc202601/data_5species/model_config/prior_bounds.json')

# theta index -> human label (mirrors attractor_analysis.theta_to_matrices)
LBL = {0: "A:So-So", 1: "A:So-An", 2: "A:An-An", 3: "b:So", 4: "b:An",
       5: "A:Vd-Vd", 6: "A:Vd-Fn", 7: "A:Fn-Fn", 8: "b:Vd", 9: "b:Fn",
       10: "A:So-Vd", 11: "A:So-Fn", 12: "A:An-Vd", 13: "A:An-Fn",
       14: "A:Pg-Pg", 15: "b:Pg", 16: "A:So-Pg", 17: "A:An-Pg",
       18: "A:Vd-Pg", 19: "A:Fn-Pg"}
A_IDX = [0, 1, 2, 5, 6, 7, 10, 11, 12, 13, 14, 16, 17, 18, 19]  # 15 interaction params
PG_EDGE = [16, 17, 18, 19]                                       # *-Pg gating candidates
STATES = ['CS', 'CH', 'DS', 'DH']
STRAT = {'CS': 'Commensal_Static', 'CH': 'Commensal_HOBIC',
         'DS': 'Dysbiotic_Static', 'DH': 'Dysbiotic_HOBIC'}


def main():
    pj = json.load(open(PRIOR_JSON))

    def prior_std(cond, i):
        b = pj['strategies'][STRAT[cond]]['bounds'].get(str(i), pj['default_bounds'])
        return (b[1] - b[0]) / np.sqrt(12.0)  # std of Uniform[l,u]

    samples = {c: paper_5sp_samples(c) for c in STATES}
    contr = {c: {i: 1 - (samples[c][:, i].std(ddof=1) ** 2) / (prior_std(c, i) ** 2)
                 for i in range(20)} for c in STATES}

    print("=== (2) posterior contraction  C = 1 - Var_post/Var_prior ===")
    for c in STATES:
        n = sum(1 for i in A_IDX if contr[c][i] > 0.5)
        print(f"  {c}: interaction params data-informed (C>0.5): {n}/15")
    print(f"\n  {'param':10s} " + " ".join(f"{c:>6s}" for c in STATES))
    for i in A_IDX:
        print(f"  {LBL[i]:10s} " + " ".join(f"{contr[c][i]:6.2f}" for c in STATES))

    print("\n=== (3) gating edge: *-Pg interactions (mean [95% CI] C) ===")
    for i in PG_EDGE:
        cells = []
        for c in STATES:
            s = samples[c][:, i]
            lo, hi = np.percentile(s, [2.5, 97.5])
            cells.append(f"{c}:{s.mean():+.2f}[{lo:+.2f},{hi:+.2f}]C={contr[c][i]:.2f}")
        print(f"  {LBL[i]:9s} " + "  ".join(cells))

    print("\n  gating score = |mean_DH| * C_DH (commensal CI should include 0):")
    scored = []
    for i in PG_EDGE:
        dh = samples['DH'][:, i]
        comm0 = all(np.percentile(samples[c][:, i], 2.5) <= 0 <= np.percentile(samples[c][:, i], 97.5)
                    for c in ('CS', 'CH'))
        scored.append((abs(dh.mean()) * contr['DH'][i], LBL[i], dh.mean(), contr['DH'][i], comm0))
    for sc, lbl, m, c, comm0 in sorted(scored, reverse=True):
        print(f"    {lbl:9s} score={sc:.3f}  mean_DH={m:+.2f} C_DH={c:.2f} commensal~0={comm0}")


if __name__ == '__main__':
    main()
