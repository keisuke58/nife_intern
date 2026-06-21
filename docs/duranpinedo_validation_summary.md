# Duran-Pinedo cross-cohort validation — internal summary

**Status: archived, not in thesis main text.**  
Mention in Discussion as: "preliminary cross-cohort evidence; full characterisation is limited by
dataset size and design gap (see Supplementary)."

---

## What was tested

Cross-cohort sign agreement between the Dieckow 2024 peri-implant 16S cohort (PRJEB71108,
n=10 patients, 3 weeks directional succession) and the Duran-Pinedo 2021 periodontal cohort
(PRJNA725874, n=15 patients, 7 biweekly time points, slow fluctuating progression).
Both cohorts were fit with the same gLV/Hamilton framework (no prior, θ=0.1 strong-pair
threshold) and the sign consistency of the strong off-diagonal pairs was compared.

## Key numbers

| Test | Result |
|---|---|
| Strong-pair sign agreement (θ=0.1, n=9 pairs) | **8/9 = 89%** |
| Within-Dieckow seed reproducibility (5 seeds, 10 seed-pairs) | **100%** sign consensus |
| Within-Duran-Pinedo seed reproducibility (3 seeds) | **100%** sign consensus |
| Shared-A penalty — Duran-Pinedo | +0.24% RMSE vs separate fit |
| Shared-A penalty — Dieckow | +12.4% RMSE vs separate fit |
| Design gap: per-step turnover | 0.28 vs 0.26 (p=0.93, same) |
| Design gap: directionality slope | 0.134 vs 0.029 (p<0.001, different) |
| Design gap: net-displacement ratio | 0.51 vs 0.19 (p=0.001, different) |
| Trajectory prediction (Dieckow A → DP) | RMSE 0.121, p=0.22 → **excluded** |
| Facilitation/competition typed transfer (Fisher) | p=0.56 → underpowered |

## Interpretation

The 89% strong-pair sign agreement is **not a fit-noise artefact**: both cohorts' signs are
fully reproducible across random seeds (100%), so the single disagreement is a genuine
biological or design-gap difference, not numerical instability.

A single shared interaction matrix explains Duran-Pinedo at nearly zero cost (+0.24%) but
costs Dieckow +12.4% — the asymmetry reflects that Duran-Pinedo dynamics are b-dominated
(slow fluctuations; A is weakly constraining), so the "+0.24%" alone is not strong evidence
of shared interactions. The more informative number is the Dieckow penalty.

The design gap is real and quantified (V1): same per-step turnover (community replacement
rate), but Dieckow is directional early-colonisation succession while Duran-Pinedo fluctuates
around a slowly progressing state. This explains why trajectory transfer fails (p=0.22) while
sign agreement holds.

## Files

- Scripts: `scripts/analysis/verify_*.py`, `strat_transfer_metabolic.py`,
  `scripts/figures/fig_duranpinedo_{backbone,network}.py`
- Results: `results/duranpinedo_validation/`
- Cross-cohort verification write-up: `docs/cross_cohort_verification_2026-06-15.md`
