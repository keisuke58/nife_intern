# Cross-cohort modelling — Outlook

Future directions beyond the limited cross-cohort claim (see
`docs/cross_cohort_verification_2026-06-15.md` for the verification battery).
These are **out of the main text**; they are thesis-Outlook / Keio-continuation
seeds. Two are prototyped here; the full versions need PBS / denser data.

## ⑤ Hierarchical Bayesian cross-cohort gLV (the principled gap-closer)

**Idea.** Instead of fitting each cohort separately and comparing signs, model the
design gap directly:
- a **shared** interaction matrix `A_shared` drawn from a hyperprior,
- per-cohort **deviations** `δ_c` with a *learned* scale `τ` (partial pooling):
  `A_c = A_shared + δ_c`, `δ_c ~ Normal(0, τ)`.

`τ → 0` collapses to one common matrix (full pooling); large `τ` recovers separate
fits. The posterior over `τ` and `A_shared` then **quantifies** how much of the
interaction structure is universal vs cohort-specific — the honest version of "the
cohorts share a backbone", with uncertainty, replacing a fragile 9-pair sign count.

**Why it fits the project.** This is the natural multi-cohort extension of the
existing TMCMC gLV/Hamilton machinery and directly operationalises the
ecological–metabolic *complementarity* advocated by [[vandenBerg2022]]: `A_shared`
can carry the AGORA sign prior while `δ_c` absorbs context.

**Implementation.** PyMC (NUTS) or the existing TMCMC driver, dispatched to PBS
(`jobs/`), not the login node — the joint fit over both cohorts' patients is heavy.

**Deterministic prototype (this repo, `scripts/analysis/outlook_hierarchical_glv.py`).**
A ridge stand-in: `A_c = A_shared + δ_c`, `δ_c` penalised by `λ_dev` (sweeps
separate↔shared), fast RK4 + L-BFGS. Results (`results/duranpinedo_validation/outlook_hierarchical_glv.json`):

| `λ_dev` | RMSE Dieckow | RMSE DP | ‖δ_dk‖ | ‖δ_dp‖ | A_dk~A_dp strong signs |
|---|---|---|---|---|---|
| 0.0 (separate) | 0.0529 | 0.0600 | **5.25** | **10.10** | 22/25 = 88% |
| 0.003 | 0.0558 | 0.0633 | 0.04 | 0.09 | 18/18 |
| 0.03 | 0.0559 | 0.0634 | 0.003 | 0.01 | 20/20 |
| 1.0 (forced shared) | 0.0583 | 0.0648 | ~0 | ~0 | 12/12 |

**Reading.** At `λ_dev=0` the cohort deviations are huge (‖δ‖ = 5.3, 10.1) yet buy
only a small RMSE gain; a *tiny* penalty (`λ_dev=0.003`) collapses the deviations to
≈0 (‖δ‖ = 0.04, 0.09) at a ~5–6% RMSE cost. So most of the apparent cohort-specific
structure is **overfitting**, not real — once mildly regularised, **one shared
interaction matrix explains both cohorts** (the shared backbone has 23 strong pairs;
its Actinobacteria/Bacilli rows match the headline interactions). This corroborates
V4 (`verify_shared_A`) with an explicit separate↔shared knee.

*Caveat:* this 550-parameter fit at capped iterations does not reach the exact
per-cohort optima (absolute RMSEs sit slightly above the separate fits in V4), so
read the **trend** (δ → 0 at mild λ ⇒ common backbone), not the absolute values. A
full hierarchical Bayesian fit would replace `λ_dev` with a learned `τ` and give
`A_shared` credible intervals.

Read it as a proof-of-concept: the shared backbone exists and the deviation norms
show which cohort departs from it; a full Bayesian fit would attach credible
intervals to `A_shared` and `τ`.

## ⑥ Independent causal cross-check (needs denser time series)

[[Ona2025]] stresses that no single inference method suffices — corroborate the
gLV-inferred *directions* with an independent, non-gLV causal method:
- **Convergent Cross Mapping (CCM)** or **Granger causality** on the guild series.

**Status: not feasible on the current data** — Dieckow has 3 timepoints and
Duran-Pinedo 7; reliable CCM/Granger needs ~20+. Outlook = apply to a future
denser longitudinal cohort (e.g. higher-frequency peri-implant sampling), where a
gLV-vs-CCM agreement would validate directionality independently of the model form.

## ⑦ Mechanistic dynamic coupling (van den Berg complementarity, full form)

Beyond sign priors: couple the ecological gLV to **dynamic FBA (COMETS)** so the
interaction coefficients are *emergent* from metabolite exchange rather than only
sign-constrained by it. The `comets/` pillar already has the AGORA-calibrated dFBA;
the Outlook is to close the loop (FBA fluxes → time-varying `A(t)`), the
"capitalise on complementarity" call of [[vandenBerg2022]].
