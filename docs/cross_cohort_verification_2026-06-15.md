# Cross-cohort claim — verification battery (2026-06-15)

Prompted by Szafranski's 2026-06-14 review (limit the cross-cohort claim;
acknowledge the Dieckow vs Duran-Pinedo design gap) and the two interaction-
inference reviews ([[vandenBerg2022]], [[Ona2025]]). Goal: stress-test the
prior-free Dieckow × Duran-Pinedo strong-pair sign agreement (8/9 = 89%) and
quantify the design gap. All on existing data; light local compute (no PBS/GPU).

Scripts: `scripts/analysis/{verify_design_difference,verify_seed_identifiability,
verify_shared_A,strat_transfer_metabolic}.py`.
Outputs: `results/duranpinedo_validation/verify_*.json`, `strat_transfer_metabolic.*`.

## Summary table

| # | Test | Result | Verdict |
|---|------|--------|---------|
| V1 | Design-difference metrics | turnover same (0.28 vs 0.26, p=0.93); **directionality slope 0.134 vs 0.029 (p<0.001)**; net-disp ratio 0.51 vs 0.19 (p=0.001) | Gap is real & quantified: Dieckow=directional succession, DP=fluctuation around slow progression |
| V2 | Within-Dieckow seed reproducibility (5 seeds) | **100%** strong-pair sign agreement (10/10 seed-pairs; 20/20 pairs identifiable) | Signs are practically identifiable; cross-cohort 89% is meaningful (ceiling=100%) |
| V3 | Duran-Pinedo multi-seed (3 seeds) | **100%** strong-pair sign consistency | DP also identifiable → comparison is ensemble-vs-ensemble |
| V4 | Shared-A pooled fit (one A, both cohorts) | DP RMSE **+0.2%**, Dieckow RMSE **+12.4%** vs separate fits | Partial common backbone (works for DP; modest cost for Dieckow) — with caveat below |
| V5 | Causal cross-check (CCM/Granger) | **not feasible** (Dieckow 3 tp, DP 7 tp) | Documented infeasible; needs longer series |
| ✗ | Stratify transfer by interaction type / metabolic set | facilitation 4/4 vs competition 4/5 (Fisher p=0.56); metabolic split degenerate | **Underpowered** — do not claim typed transfer (see `strat_transfer_metabolic`) |

## What this means for the claim

- The cross-cohort agreement is **not a fit-noise artefact**: both cohorts' signs are
  fully reproducible across random seeds (V2, V3), so the single 1/9 disagreement is a
  genuine cross-cohort difference.
- The **design gap is real and now numerical** (V1): same per-step turnover, but Dieckow
  is directional (early colonization) while Duran-Pinedo fluctuates around a slowly
  progressing state. This is exactly why the claim must stay *coarse / strong-pair sign*.
- A **single shared interaction matrix explains Duran-Pinedo almost perfectly (+0.2%)** and
  Dieckow at a modest cost (+12.4%) (V4) — the principled version of "a common backbone
  exists", stronger than a 9-pair count.

### Honest caveats
- V4 asymmetry + b-domination: Duran-Pinedo's near-zero shared-A penalty partly reflects
  that its **slow dynamics are b-dominated** (A is only weakly constraining), so "+0.2%" is
  not by itself strong evidence of shared *interactions*. The Dieckow +12.4% (the more
  A-driven cohort) is the more informative number — a shared A is meaningfully worse than
  Dieckow's own, i.e. some interaction structure is cohort-specific.
- The typed-transfer stratification is underpowered (n too small); we therefore make **no**
  mechanistic "cross-feeding transfers, competition doesn't" claim at the cross-cohort level
  (that distinction is supported only *within* Dieckow, by the AGORA permutation test).
- V4 uses fast fixed-step RK4 (not the canonical adaptive integrator); numbers are for the
  internal shared-vs-separate comparison, not to replace the canonical fits.

## Bottom line
The verifications **support keeping the cross-cohort result as a limited, honest claim**
(coarse strong-pair sign agreement that is reproducible and rests on a common backbone),
while the design gap and the underpowered stratification justify *not* overstating it —
consistent with the deck's tone-down. Principled gap-closing (hierarchical shared+deviation
gLV; longer-series causal methods) remains Outlook.
