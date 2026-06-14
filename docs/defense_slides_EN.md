---
title: "Metabolism Constrains Oral-Biofilm Community Dynamics"
subtitle: "Ecological ODE inference, a metabolic sign prior, and a spatial extension — Master's defense"
author: "Keisuke Nishioka — NIFE"
date: "2026-11"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
  - \newcommand{\relu}[1]{\left[#1\right]_{+}}
---

## Research objectives and overview

We model the **dysbiosis of the oral-biofilm community** relevant to
peri-implantitis as the time evolution of community composition
(SIIRI consortium).

- Clinical problem: a microbial community that transitions from commensal to
  dysbiotic around titanium implants.
- Research question: **what drives this transition** — and what does
  metabolism, computed from genomes, tell us about ecological interactions?
- Central thesis: **metabolism constrains the sign of ecological interactions.**

\vspace{0.4em}
This defense presents ecological ODE inference, a metabolic sign prior, and a
spatial extension as a coherent research arc.

---

## Graphical abstract

![](results/figures/concept_overview_pub.png){ height=78% }

Metabolism fixes the **sign** of interactions; dysbiosis appears as a **re-wiring
+ spatial re-organisation** (P. gingivalis centralises and sinks deep).

---

## The research arc

The work has two stages:

1. **Original (first half)** — a 5-species in-vitro system, inferred by
   **GPU Bayesian ODE inference**. TMCMC ($10^4$ particles) recovered the
   posterior for four attractors **CS / CH / DS / DH**
   (commensal/dysbiotic × static/HOBIC).
2. **Extension (second half)** — carry the framework to real data:
   a **metabolic AGORA sign prior** + **in-vivo longitudinal 16S** +
   a **spatial PDE** + **FISH**.

\vspace{0.4em}
The first half establishes the dynamical-systems skeleton (attractors, pH
prediction); the second grounds it in real data via metabolic constraints and
spatial structure.

---

## Core model (concise)

Generalized Lotka–Volterra for absolute abundances $x_i$:
$$\dot{x}_i = x_i\Big(b_i + \sum_{j=1}^{S} A_{ij}\,x_j\Big),\qquad S=10 .$$

Replicator / Hamilton form on the simplex $\sum_i\varphi_i=1$
(16S is compositional):
$$\dot{\varphi}_i = \varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big].$$

**Sign prior:** from AGORA cross-feeding flux $F_{ij}$ build
$$P_{ij}=\sgn(F_{ij})\in\{-1,0,+1\},$$
constraining only the **sign** of $A$ (magnitude discarded by design).

---

## Data flow (real data)

![](results/figures/pipeline_overview_pub.png){ height=66% }

Real Dieckow data: guild $\varphi$ (10 patients × 3 wk) → AGORA sign prior →
gLV/Hamilton fit → LOO-CV → the four attractors.
\texttt{fit\_*.json} is the interchange format.

---

## AGORA sign-prior pipeline

![](results/fig2_agora_pipeline.png){ height=64% }

(A) the procedure; (B) adding layers grows the constraint set $|\mathcal{M}|$
from $10\to22\to58$ (+36 from AGORA L3); (C) the resulting sign-prior matrix
$P=\sgn(F)$.

---

## Empirical sign agreement (naive estimate)

![](results/fig3_agora_sign_validation.png){ height=55% }

$$\mathrm{SA}=\frac{1}{|\mathcal{M}|}\sum_{(i,j)\in\mathcal{M}}\mathbb{1}\!\left[\sgn(\hat A_{ij})=P_{ij}\right]=92\%.$$
\textcolor{red}{This overstates it — the prior is sign-degenerate (honest test next).}

---

## Honest validation

The naive 92% is inflated by sign degeneracy. The correct control is a
permutation test:

- Only the **cross-feeding direction** is independently validated:
  $$p=4\times10^{-4}\quad(z=+3.79,\ n_\pi=10^4).$$
- The **competition direction is NOT validated** (chance level).
- Across **two cohorts** (Dieckow × Duran-Pinedo), prior-free, strong-pair signs
  agree at **89%** ($p\approx0.02$).

\vspace{0.3em}
\textcolor{red}{The AGORA prior is not reproduced by the 16S dynamics — it is a
modelling choice, not a data-confirmed fact.}

---

## Network view

Structural analysis of the inferred $A$ (class level):

- **Veillonella (Negativicutes) is the metabolic sink** — the receiver that
  aggregates lactate and other products.
- The classic **Pg-keystone / Fn-bridge picture is NOT supported at class level.**
- In dysbiosis **Pg centralises** (eigenvector centrality $0.32\to0.51$).
- At the same time the **S. oralis–Veillonella mutualism breaks.**

\vspace{0.4em}
Dysbiosis is not a mere compositional change but a **rewiring** of the
interaction network.

---

## Spatial extension — FISH depth profiles (CH vs DH)

![](results/diffusion_fit/zprofiles_all_ti_overlay.png){ height=50% }

Depth-resolved 5-species profiles from CLSM-FISH. The depth structure differs
between commensal-HOBIC (CH) and dysbiotic-HOBIC (DH).

---

## Dysbiosis = spatial reorganisation

![](results/diffusion_fit/depth_niche.png){ height=52% }

Dysbiosis is more than a compositional shift — it is a **spatial
reorganisation**: *P. gingivalis* sinks deep (**+30 µm** from day 6, the
anaerobic niche).

---

## NEW: a spatial cross-feeding test

![](results/diffusion_fit/spatial_crossfeeding.png){ height=48% }

The lactate pair **S. oralis (producer) → Veillonella (consumer)** is spatially
stratified: Veillonella is **shallower** (10/10 samples, Wilcoxon $p=0.002$).
Producer basal, consumer above — consistent with **upward lactate diffusion**.

---

## Reaction–diffusion PDE

Extend the 0D gLV to depth-resolved reaction–diffusion along $z$:
$$\partial_t\varphi_i = D_i\,\partial_z^2\varphi_i - u\,\partial_z\varphi_i + R_i(\varphi),$$
$R_i$ the replicator reaction term (with $A,b$ fixed from the temporal fit);
only $D_i$ (diffusion) and $u$ (advection) are inferred from the depth profiles.

\vspace{0.4em}
\textcolor{red}{This is preliminary — the HPC parameter sweep is converging.}
The spatial fit grounds the model on an observable (FISH depth) independent of
the temporal dynamics.

---

## pH prediction (original 5-species work, independent validation)

![](results/figures/fig_ph_validation.png){ height=52% }

Predict pH forward from the 5-species attractor posterior:
independent $R^2=0.78$, RMSE $0.13$, LOO $R^2=0.92$.
**pH is not used in calibration** — a genuinely independent validation.

---

## Conclusions

1. **Metabolism constrains the sign of ecological interactions**
   ($P_{ij}=\sgn(F_{ij})$; magnitude is not used).
2. Dysbiosis is a **rewiring + spatial reorganisation**, with
   *P. gingivalis* sinking deep (+30 µm).
3. That claim reproduces across **two cohorts** (Dieckow × Duran-Pinedo, 89%,
   $p\approx0.02$) and an **independent mechanistic route** (COMETS dFBA).
4. The original 5-species ODE predicts pH independently
   ($R^2=0.78$, LOO $R^2=0.92$).

---

## Outlook

**Near term**

- **Neutral-init LOO** (submitted) — confirm $p=4\times10^{-4}$ sign enrichment is data-driven, not warm-start artefact.
- **Unified inference** — TMCMC posteriors regularised by metabolic sign prior → credible intervals on in-vivo $A_{ij}$.
- **Asymmetric $A$ channel** — relax $A_{ij}=A_{ji}$ for directional metabolite pairs; test whether competition direction recovers significance.

**Medium term**

- **Fn knock-out prediction** — omitting *F. nucleatum* should collapse the dysbiotic *Pg* surge (direct co-culture test).
- **Full time-series PINN fit** — turn placeholder $D_i$ into measured diffusivities; couple $O_2$-PDE to Monod kinetics.

**Long term**

- **Clinical dysbiosis index** — 10-guild projection onto commensal–dysbiotic axis, calibrated from attractor states.
- **Patient-specific MAGs** — replace AGORA2 reference strains with personalised GEMs for tighter $A_{ij}$ priors.

\vspace{0.3em}
\textit{Same principle throughout: mechanistic structure makes scarce data yield testable models.}

---

## Take-home

\begin{center}
\large
Metabolism constrains the \textbf{sign} of ecological interactions.\\[0.3em]
Dysbiosis is a \textbf{rewiring + spatial reorganisation}.\\[0.3em]
Reproduced across \textbf{two cohorts}, an \textbf{independent dFBA},
and a \textbf{held-out pH prediction}.
\end{center}

\vspace{0.6em}
One **10-guild taxonomy** and the **`fit_*.json`** format tie it all together.
