# Idea — design the implant to break the mechano-biological vicious cycle (2026-06-20)

A novel extension that turns the project's *descriptive* model (when does peri-implantitis tip into
runaway loss?) into a *prescriptive* one (what implant geometry keeps the patient below the tipping
point?). The research significance is intrinsic: it is the first design objective derived *through* a
coupled biofilm–inflammation–mechanics disease model, and it yields a concrete clinical artefact (a
geometry, or a per-patient geometry recommendation). The computational-mechanics methods it later needs
(topology optimisation, phase-field, ML surrogates) are a means to that end, not the motivation.

## The one-line idea
The vicious cycle ignites when the crest enters Frost's pathological-overload window
($>3000~\mu\varepsilon$). So **optimise the implant geometry such that the crestal strain stays below
that window across the entire marginal-bone-loss trajectory ($L: 0\!\to\!4$ mm)** — mechanically
*opening* the feedback loop instead of merely describing it.

## Why it is novel
Conventional implant FEA searches for "a design with low stress" at a single bone level. Here the
objective is defined **through the vicious-cycle model**: not "minimise stress" but "**maximise the
dysbiosis threshold $b_{\mathrm{crit}}$ at which the loop tips**" — i.e. design for the largest
biological insult the implant can mechanically tolerate before runaway loss. Biology × mechanics ×
optimisation in one objective; the payoff is a concrete clinical artefact (a geometry, or a per-patient
geometry recommendation), not just an explanation.

## Formulation
- **Design variables** $\theta$: implant diameter $D$, length, thread pitch & depth, taper, neck /
  platform-switch geometry, abutment/crown moment arm. (All already parameterised in
  `implant_coupon.py` / `run_coupons.sh`, `run_crown_design.sh`.)
- **Inner model**: FEM → crestal strain field $\varepsilon(\theta, L)$ across a bone-loss sweep $L$;
  feed the resulting $A_\theta(L)$ (stress amplification vs loss) into the RANKL/OPG vicious-cycle ODE
  → tipping threshold $b_{\mathrm{crit}}(\theta)$ and 36-mo loss distribution.
- **Objective**: $\max_\theta\; b_{\mathrm{crit}}(\theta)$  (equivalently, minimise the fraction of the
  $L$-trajectory spent in the $>3000~\mu\varepsilon$ window).
- **Constraints**: primary stability (micromotion < 50–150 µm), osseointegration strain floor (don't
  stress-shield: keep crest above the disuse window), manufacturability, ISO-14801 fatigue margin.
- **Robust / patient-specific variant**: optimise $b_{\mathrm{crit}}$ under the GDI / parameter
  uncertainty already quantified by the UQ (Monte-Carlo) — "design for the 90th-percentile patient".

## Maturity tiers (what is reachable when)
- **T0 — parametric (reachable now).** The coupon design sweep already exists
  (`coupon_results.jsonl`, `design_results.jsonl`): crestal p95 vs $D$/$L$/pitch/taper/crown height.
  Re-score each existing design by its $b_{\mathrm{crit}}$ instead of raw stress → first ranking.
- **T1 — surrogate + Bayesian optimisation.** GP surrogate of $b_{\mathrm{crit}}(\theta)$ over the
  design box + BO (the exact GP+BO machinery from the SiC-laser track transfers). ~dozens of FEM
  evaluations. A defensible *optimised* design.
- **T2 — topology / shape optimisation (Keio core).** Free-form neck/thread shape via adjoint or
  level-set/phase-field; couples to the existing phase-field UEL (`uel_phasefield_at2.f`) for
  interface damage. Muramatsu-aligned.
- **T3 — robust patient-specific.** Per-GDI-class optimal geometry under uncertainty; ML / quantum-
  annealing surrogate for the inverse "which design for which patient" map.

## Existing assets it reuses
- Parametric implant builder + design sweeps (`implant_coupon.py`, `run_coupons.sh`, `run_crown_design.sh`).
- FEM crestal-strain metric + Frost window (`bone_standard_laws.py`, `bone_remodeling_fem_field.py`).
- Vicious-cycle ODE + tipping + UQ (`fig_periimplantitis_rankl_opg.py`, `fig_periimplantitis_uq.py`).
- GP + Bayesian-optimisation experience from the SiC-laser process track.

## Honest scope
- **Current thesis**: one Outlook sentence ("the descriptive tipping model is directly invertible into
  a design objective $\max_\theta b_{\mathrm{crit}}$; T0 re-scoring of the existing design sweep is a
  natural next step"). Do NOT over-build it here.
- **Beyond the thesis**: the value stands on its own (a mechanism-grounded implant-design tool); the
  computational-mechanics depth (topology optimisation under a biological objective) is a natural place
  to take it further, optionally aligning with a continuum-mechanics group. See
  [[project_continuum_mechanics_bridge]].
- Caveat: the inner vicious-cycle numbers are calibration-dependent (model); a real optimisation needs
  the biological priors tightened (the value-of-information / PICF measurement loop), so the design and
  the biomarker-measurement recommendations reinforce each other.
