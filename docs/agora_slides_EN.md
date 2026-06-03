---
title: "Integrating Metabolism and Ecology through AGORA"
subtitle: "From genome-scale metabolic models to interaction signs — formal treatment"
author: "Keisuke Nishioka — NIFE / SFB TRR-298"
date: "2026-06-03"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
  - \newcommand{\relu}[1]{\left[#1\right]_{+}}
---

## Where this deck sits

The project = three pillars + a spatial extension. Data flow:

raw 16S → guild $\varphi$ → gLV/Hamilton (+ sign prior) → LOO validation → spatial PDE

The decks (you-are-here in **bold marker**):

- **Overview** (umbrella) — the whole picture, three pillars
- **AGORA** — metabolism → sign prior (input to the ecological model)  — **(you are here)**
- **Dieckow** — in-vivo interaction inference & validation (the model)
- **Network** — structural analysis of the interaction matrix $A$
- **Spatial-PDE** — reaction-diffusion of the FISH depth profiles
- **FISH pipeline** — .lif → 5-species depth composition

---

## Scope and claim

Constrain an **ecological interaction matrix** with **metabolism computed from
genomes (AGORA2)**, and quantify how far that constraint is empirically supported.

- the ecological model: generalized Lotka–Volterra (gLV) / replicator
- the metabolic engine: flux-balance analysis (FBA, pFBA, MICOM)
- the bridge: a **sign prior** $P_{ij}=\sgn(F_{ij})$ derived from cross-feeding flux
- the validation: permutation test on the prior-free fit, two-cohort replication

\vspace{0.4em}
**Thesis.** The metabolic signal constrains the **sign** of cooperative
interactions (not the magnitude, not competition), with $p=4\times10^{-4}$.

---

## The ecological model

Generalized Lotka–Volterra for absolute abundances $x_i$:
$$\dot{x}_i = x_i\Big(b_i + \sum_{j=1}^{S} A_{ij}\,x_j\Big),\qquad i=1,\dots,S=10 .$$

On the simplex $\varphi_i = x_i/\sum_k x_k$ (compositional 16S data), the
replicator / Hamilton form is
$$\dot{\varphi}_i = \varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big],
\qquad \sum_i \varphi_i = 1 .$$

- $A_{ij}>0$: $j$ **promotes** $i$;  $A_{ij}<0$: $j$ **suppresses** $i$.
- Inference is **underdetermined**: $\mathcal{O}(S^2)=100$ parameters vs
  $10$ patients $\times\,3$ weeks. The signs of $A$ are not identifiable from
  abundance alone $\Rightarrow$ we add a **metabolic sign prior**.

---

## Flux-balance analysis (FBA) as a linear program

A genome-scale model gives a stoichiometric matrix $S\in\mathbb{R}^{m\times n}$
(metabolites $\times$ reactions). Steady-state growth is the LP
$$\max_{v}\; c^{\!\top} v \quad\text{s.t.}\quad S v = 0,\;\; v^{-}\le v \le v^{+},$$
with $c$ selecting the biomass reaction ($\mu = c^{\!\top}v$). **Parsimonious FBA**
removes flux loops by a second stage,
$$\min_{v}\; \sum_{j}\lvert v_j\rvert \quad\text{s.t.}\quad c^{\!\top}v=\mu^{\star},\; Sv=0,\; v^{-}\le v\le v^{+}.$$

Exchange fluxes give per-guild **secretion** $s_{j\alpha}=\relu{v^{\text{ex}}_{j\alpha}}$
and **uptake** $u_{i\alpha}=\relu{-v^{\text{ex}}_{i\alpha}}$ over metabolites $\alpha$.

---

## What AGORA2 supplies

**AGORA2** (Heinken et al., *Nat. Biotechnol.* 2023, 41:1320): **7,302**
genome-scale reconstructions, extended from gut to **oral** taxa; one
representative SBML model per guild.

- representatives: Bacilli = *S. gordonii*, Negativicutes = *V. parvula*,
  Bacteroidia = *P. melaninogenica*, Fusobacteriia = *F. nucleatum*, …
- oral-fluid medium (Dawes 2008): sugars, 20 amino acids, B-vitamins, trace
  metals, cell-wall precursors $\Rightarrow$ positive growth for all 10 guilds
  ($\mu = 0.11\text{–}1.66\,\mathrm{h^{-1}}$).

\vspace{0.3em}
Medium design is a genuine sensitivity: a too-poor medium drives $\mu\to0$ and
collapses the cross-feeding signal.

---

## Cross-feeding score $\to$ sign prior

Let $\mathcal{T}=\{\text{H}_2\text{O}_2,\text{H}_2\text{S}\}$ be toxins. Define a
directed **net metabolic flow** from $j$ to $i$,
$$F_{ij} \;=\; \underbrace{\sum_{\alpha\notin\mathcal{T}} w_\alpha\, s_{j\alpha}\,u_{i\alpha}}_{\text{cross-feeding }(+)}
\;-\; \underbrace{\sum_{\alpha\in\mathcal{T}} w_\alpha\, s_{j\alpha}\,u_{i\alpha}}_{\text{toxin }(-)} .$$

The **sign prior** keeps only the direction:
$$P_{ij} = \sgn(F_{ij}) \in \{-1,\,0,\,+1\}.$$

Magnitude is discarded by design — FBA flux units $\neq$ ecological units.
Across $10\times 9=90$ directed pairs this yields the constrained set
$\mathcal{M}=\{(i,j): P_{ij}\neq 0\}$.

---

## The pipeline at a glance

![](results/fig2_agora_pipeline.png){ height=64% }

(A) procedure; (B) $|\mathcal{M}|$ grows $10\to22\to58$ across layers (+36 from
AGORA L3); (C) the resulting sign-prior matrix $P=\sgn(F)$.

---

## Bayesian integration: the sign-prior penalty

The fit minimises trajectory error plus a one-sided hinge that penalises only
**sign violations** (never magnitude):
$$\mathcal{L}(A,b) = \frac{1}{2\sigma^{2}}\sum_{t}\big\lVert \varphi^{\text{obs}}_{t}-\varphi^{\text{pred}}_{t}(A,b)\big\rVert^{2}
\;+\; W\!\!\sum_{(i,j)\in\mathcal{M}}\!\! \relu{-\,P_{ij}\,A_{ij}} .$$

- $\relu{-P_{ij}A_{ij}}=\max(0,-P_{ij}A_{ij})$ is $0$ when $\sgn(A_{ij})=P_{ij}$.
- $W$ = prior stiffness; $\sigma$ = observation scale.
- Posterior sampled by **TMCMC** ($10^4$ particles) for the 5-species attractors;
  guild-level fits by L-BFGS-B / TMCMC.

---

## The three evidence layers

| Layer | Source | $w_\alpha$ | Basis |
|---|---|---|---|
| **L1** | Szafrański Suppl. (experimental + KEGG/HMDB) | 2.0 | direct |
| **L1** | Szafrański Suppl. (experimental, unannotated) | 1.5 | direct |
| **L2** | Szafrański Suppl. (predicted) | 1.0 | prediction |
| **L3** | **AGORA2 pFBA cross-feeding** | 1.0 | genome-scale |

$w_\alpha=\max$ over rows of metabolite $\alpha$. Canonical example — **lactate**:
$$\text{Bacilli (Strep)} \xrightarrow{\ \text{lactate}\ } \text{Negativicutes (Veillonella), Actinobacteria}.$$

---

## Why magnitude priors fail — the MacArthur view

Consumer–resource theory (MacArthur 1970; Marsland 2019) derives
$$A_{ij} = \underbrace{\sum_{\alpha} s_{j\alpha}\,c_{i\alpha}}_{\text{cross-feeding}\,(+)}
\;-\; \underbrace{\frac{c_i\!\cdot\! c_j}{\lVert c_i\rVert\,\lVert c_j\rVert}}_{\text{niche overlap}\,(-)} .$$

- **Niche-overlap (cosine) term saturates**: oral taxa are generalists, so
  $\cos(c_i,c_j)\approx1$ for nearly all pairs $\Rightarrow$ everything reads as
  competition (empirically $6$ positive / $84$ negative pairs — useless).
- Growth-rate-suppression priors fail identically in poor media.

$\Rightarrow$ keep $\sgn(A_{ij})$, discard the magnitude.

---

## Empirical sign agreement (the naive estimate)

![](results/fig3_agora_sign_validation.png){ height=55% }

$$\mathrm{SA}=\frac{1}{|\mathcal{M}|}\sum_{(i,j)\in\mathcal{M}}\mathbb{1}\!\left[\sgn(\hat A_{ij})=P_{ij}\right]=\frac{66}{72}=92\%.$$
\textcolor{red}{This overstates support — the prior is sign-degenerate (next slide).}

---

## MICOM — community FBA

Single-species pFBA tests only **feasibility** ($j$ *can* secrete $X$, $i$ *can*
eat $X$). **MICOM** (Diener 2020) instead solves the joint community and checks
whether the flux *actually flows*, via a **cooperative trade-off**:
$$\max\; \min_{i}\frac{\mu_i}{\mu_i^{\max}}\quad\text{s.t.}\quad
\sum_i \mu_i \ge \tau\sum_i \mu_i^{\max},\;\; S^{\text{com}}v=0,\;\tau=0.5 .$$

- each guild is guaranteed at least a fraction $\tau$ of its maximal growth, then
  the shared flux pattern is fixed;
- even though oral taxa are generalists, **only routes carrying real community
  flux activate** — the unspecific "everyone competes" artefact disappears;
- the secretion/uptake exchange fluxes $s_{j\alpha},u_{i\alpha}$ are read off the
  community solution and fed into $F_{ij}$ exactly as before.

---

## MICOM — results

| Method | Sign agreement | $|\mathcal{M}|$ |
|---|:---:|:---:|
| Literature L1+L2 | 45% (5/11) | 11/45 |
| single-species pFBA v1 | 88% (29/33) | 33/45 |
| **MICOM (community)** | **100% (36/36)** | **36/45** |

Directly resolved lactate exchange (community flux):
$$\text{Bacilli}\xrightarrow[\;-97.9\;]{\;+97.7\;\text{mmol gDW}^{-1}\text{h}^{-1}}\text{Negativicutes}\quad(\mathrm{EX\_lac\_L}).$$
Caveat: $\hat A$ was fit under the v1 prior, so $100\%$ may reflect containment.

---

## Prior stiffness $W$ — a phase transition

![](results/fig_agora_weight_sensitivity.png){ height=55% }

At $W=1.0$: $\mathrm{SA}\to100\%$, $\mathrm{LOO\text{-}RMSE}\approx0.050$. The
prior-free gLV reaches $0.0455$, so the prior buys **interpretability and
sign-consistency**, not raw predictive accuracy — stated honestly.

---

## Critical validation: is the prior independent of the data?

At $\alpha=0$ the prior is **all-positive** (cross-feeding only), so $\mathrm{SA}$
is inflated. The correct control compares **off-prior** cells via a label
permutation, with statistic
$$z=\frac{\mathrm{SA}-\mathbb{E}_\pi[\mathrm{SA}]}{\sqrt{\operatorname{Var}_\pi[\mathrm{SA}]}},\qquad n_\pi=10^4 .$$

| Model | cross-feeding direction | competition |
|---|---|---|
| **Hamilton (symmetric), $\alpha=0$** | $\mathbf{78.6\%}$ (11/14) vs $\mathbb{E}_\pi=37.7\%$, $\;p=4\!\times\!10^{-4},\,z=+3.79$ | $\approx$ chance |
| gLV (asymmetric) | 41% (null) | null |

- **Only the cross-feeding direction is validated**; competition is not.
- The AGORA prior is **not reproduced** by the 16S dynamics $\Rightarrow$ it is a
  modelling choice, not a data-confirmed fact.
- **Two cohorts** (Dieckow $\times$ Botelho), prior-free, agree on strong-pair
  signs at $\mathbf{89\%}$ ($p\approx0.02$).

---

## Mechanistic cross-check (COMETS dynamic FBA)

The same AGORA GEMs drive a 5-species **dFBA** forward simulation (Monod-coupled
exchange), independent of the prior:

![](comets/pipeline_results/sweep_crossfeeding.png){ height=50% }

Healthy: So/An dominant, lactate cross-feeding, $\mathrm{DI}=0.15$. Diseased:
Pg/Fn expansion, $\mathrm{DI}=0.70$. The commensal$\leftrightarrow$dysbiotic split
emerges forward, corroborating the inferred interactions.

---

## Limitations and conclusions

**Limitations.** guild $=$ class (representative $\neq$ guild); $20/22$
inhibition rows are oxygen (no producer) $\Rightarrow$ only the H$_2$O$_2$ toxin
fires; $w_\alpha$ is a per-metabolite max; magnitude is discarded.

**Conclusions.**
1. $\text{AGORA pFBA}\to F_{ij}\to P_{ij}=\sgn(F_{ij})$ is the methodological novelty.
2. **Sign usable, magnitude not** (MacArthur cosine saturation avoided).
3. single-species ($92\%$) $\to$ **MICOM ($100\%$)** captures community context.
4. Honest validation: **cross-feeding only**, $p=4\times10^{-4}$; two cohorts $89\%$;
   the prior is a modelling choice.
5. COMETS dFBA reproduces the dysbiotic split forward.
