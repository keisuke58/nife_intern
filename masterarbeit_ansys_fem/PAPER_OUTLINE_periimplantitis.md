# Paper outline — A mechanistic chain from the oral microbiome to peri-implant bone loss

**Status:** skeleton / seed (2026-06-14). Built from the NIFE thesis assets; intended both as a standalone
manuscript draft and as a continuation seed for the Keio / Muramatsu phase (computational solid mechanics
× Bayesian × multiscale). Not frozen — edit freely. The thesis itself stays under the freeze policy
([[feedback_thesis_freeze]]); this document is the *paper* track, separate from the Masterarbeit body.

## Central figure
`figures/fem_graphical_abstract_polished.pdf` — the five-stage chain is the spine of the paper:
**microbial ecology → biofilm anatomy → host inflammation (tipping point) → biomechanics (vicious cycle)
→ per-patient prediction**, with the bistability/hysteresis result (`fem_periimplantitis_bistability.pdf`)
as the mechanistic punchline.

## Title options
1. *From the oral microbiome to peri-implant bone loss: a bistable mechano-immunological model of
   peri-implantitis* (recommended — names the chain and the punchline)
2. *Peri-implantitis as a tipping-point disease: coupling biofilm dysbiosis, host inflammation and
   implant biomechanics*
3. *A literature-anchored cascade model linking dysbiosis, RANKL/OPG and marginal bone loss around dental
   implants*

## Target venues (in rough fit order)
- *J. Clin. Periodontol.* / *Clin. Oral Implants Res.* — clinical-translational framing (reversibility
  window, host-modulation, prevention≪cure).
- *J. Theor. Biol.* / *PLoS Comput. Biol.* / *Math. Biosci.* — the bistability/dynamical-systems framing.
- *J. Mech. Behav. Biomed. Mater.* / *Biomech. Model. Mechanobiol.* — the FEM + mechanical-feedback framing
  (and the natural Keio/Muramatsu venue).
- Conference seed: WCCM/USNCCM/IWSHM ([[reference_conferences_2027]]) for the multiscale-coupling angle.

## One-line contribution
We assemble published mechanism into a single coupled model that (i) reproduces clinical bone-loss
trajectories from one free parameter, (ii) shows the disease is **bistable with hysteresis** once the
ecology↔host feedback loop is closed, and (iii) turns that into clinically actionable maps (reversibility
window, therapy-node comparison, implant-design effect).

## Abstract (draft, ~200 words)
> Peri-implantitis is driven by a dysbiotic submucosal biofilm, but why it resists hygiene once
> established, and how mechanical and host factors combine, remain hard to reason about quantitatively.
> We build a coupled model linking five published mechanisms: a guild-level dysbiosis index from
> longitudinal 16S data; the submucosal biofilm anatomy; a host-inflammation cascade through the RANKL/OPG
> balance to osteoclastic marginal bone loss; and a finite-element mechanical feedback in which recession
> raises crestal stress that aggravates resorption only in the presence of inflammation. Every rate is
> taken from the literature except a single bone-loss gain, pinned to reported active-phase progression
> (~3 mm over 36 months; fitted k_L = 0.022 wk⁻¹). Closing the ecology–host loop — inflammation supplies
> the heme-iron that inflammophilic anaerobes require — renders the system **bistable**: a healthy and a
> diseased attractor separated by a threshold, with hysteresis, so a transient insult locks the implant
> into progressive loss. The model yields a reversibility-window map, a host-modulation therapy-node
> comparison (anti-RANKL arrests loss downstream without removing the cause), and an implant-design
> time-to-failure coupling. It reframes peri-implantitis as a tipping-point disease and explains the
> clinical primacy of prevention over cure.

## Section skeleton (mapped to existing assets)

| § | Content | Figure / data asset | Key numbers |
|---|---|---|---|
| 1 Intro | peri-implantitis burden; gap = no quantitative chain microbiome→bone; aim | graphical abstract | — |
| 2.1 Ecology | guild dysbiosis index GDI from longitudinal 16S | Dieckow PRJEB71108, Duran-Pinedo PRJNA725874 | GDI = log(Bact+Clos+Fuso)−log(Bac+Act+Neg) |
| 2.2 Anatomy | submucosal biofilm between Ti and gingiva; descent path | `fem_transmucosal.pdf` | sulcus→infrabony |
| 2.3 Host cascade | dysbiosis→inflammation→RANKL/OPG→osteoclast→loss; mech. synergy | `fem_periimplantitis_inflammation.pdf` | Hill threshold; OC ~2 wk |
| 2.4 Mechanics | FEM crestal-stress vs bone level; coupon design sweep | `fem_periimplantitis_progression.pdf`, coupon results | crest 14→35 MPa @ 2→8 mm loss |
| 3.1 Calibration | one free k_L pinned to clinical progression; face validity | `fem_periimplantitis_calibration.pdf` | k_L=0.022 wk⁻¹; worst-case 3 mm@36mo |
| 3.2 Bistability | close ecology↔host loop → two attractors + hysteresis | `fem_periimplantitis_bistability.pdf` | hysteresis window b₀∈[0.03,0.41] |
| 3.3 Prediction + UQ | per-patient trajectories; Monte-Carlo CI; sensitivity | `fem_periimplantitis_duranpinedo.pdf`, `..._uq.pdf` | kC/gC/kL dominate; P(progress)≤0.12 |
| 3.4 Reversibility | when/how aggressively to intervene | `fem_periimplantitis_reversibility.pdf` | brushing window ~13–19 mo; debride ~22 mo |
| 3.5 Therapy nodes | biofilm vs anti-TNF vs anti-RANKL vs combo | `fem_periimplantitis_hostmod.pdf` | 36-mo loss 2.98→1.20 mm |
| 3.6 Design buys time | diameter→time-to-critical | `fem_periimplantitis_design_time.pdf` | Ø3.5/4.1/4.8 = 61/65/68 mo |
| 4 Discussion | tipping-point reframing; prevention≪cure; design secondary | — | — |
| 5 Limitations | see below | — | — |
| 6 Outlook | Keio continuation (below) | — | — |

## Honest limitations (state explicitly — these are strengths of the framing, not weaknesses to hide)
- Rates are literature-anchored, **not** fitted per-timepoint; only one parameter (k_L) is calibrated.
- The non-pathological early-MBL plateau is biologic-width remodelling — a different mechanism, **not**
  reproduced by design.
- Mechanical feedback is a **secondary** modifier (design buys ~7 mo over 5+ yr); biofilm control dominates
  — consistent with overload-needs-inflammation synergy.
- Cohorts available are sub-clinical/periodontitis (Dieckow/Duran-Pinedo), not longitudinal peri-implantitis
  with bone loss; absolute risk numbers are illustrative, relative stratification is the robust claim.
- bistability parameters are dimensionless and tuned to the bistable regime; the *qualitative* existence
  of two attractors + hysteresis is the claim, not the exact window.

## What's needed to go from skeleton → submittable
1. **Per-timepoint validation**: obtain **PRJNA1215005** (Anuntakarun 2025; longitudinal 16S + radiographic
   bone loss + PD/BOP, 7 patients × baseline/3/6 mo) → fit/validate the cascade against measured bone loss,
   turning the calibration from face-validity to a genuine Bayesian posterior.
2. **Explicit RANKL/OPG (3-variable)**: replace the lumped osteoclast node with a Lemaire/Pivonka-style
   RANK–RANKL–OPG sub-model for biological literalness and citable structure.
3. **Sensitivity/identifiability** of the bistability boundary to the loop-gain parameters.
4. Figure consolidation to journal format (thesis_style already usetex/lmodern).

## Keio / Muramatsu continuation bridge ([[project_future_keio_muramatsu]], [[project_continuum_mechanics_bridge]])
The paper's mechanical layer is deliberately a reduced feedback (crestal-stress scaling). The natural
Keio extension (computational solid mechanics; phase-field, multiscale, FEM) is to **replace the scalar
feedback with a spatially-resolved bone-remodelling field**:
- couple the inflammation cascade to a **phase-field / mechanostat bone-resorption FEM** (SED-driven
  resorption front → saucerization geometry), closing the loop between the host-response ODE and the
  evolving bone domain;
- treat the biofilm–bone interface with the existing **growth-eigenstrain + cohesive/phase-field
  detachment** machinery as a moving-boundary problem;
- carry the **Bayesian/UQ** spine into the FEM (parameter posteriors → predictive bone-loss fields with
  credible intervals) — the methodological through-line from NIFE to Keio.
This is also the WCCM/IWSHM-presentable multiscale story and the cleanest seed for the 2028 Keio thesis.

## Asset locations
- Figures + scripts: `masterarbeit_ansys_fem/figures/`, `masterarbeit_ansys_fem/extensions/fig_periimplantitis_*.py`
- FEM results: `/home/nishioka/IKM_Hiwi/FEM/tier2b_real/*.jsonl` (coupon/pimp/contact)
- Running record + literature anchors: `masterarbeit_ansys_fem/coupling_prototype/tier2b_real/README.md`
- Data: `data/prjna725874/` (Duran-Pinedo), Dieckow guild abundances under `results/dieckow_cr/`
