---
title: "Inferring Oral-Microbiome Interactions In Vivo (Dieckow 16S)"
subtitle: "From compositional time-series to a signed interaction matrix $A$ — fit, LOO-CV, critical validation"
author: "Keisuke Nishioka — NIFE"
date: "2026-06-11"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
  - \newcommand{\relu}[1]{\left[#1\right]_{+}}
---

## Position within the project

The project comprises three pillars and a spatial extension. Data flow:

raw 16S → guild $\varphi$ → gLV/Hamilton (+ sign prior) → LOO validation → spatial PDE

Decks in the series (the present deck in **bold**):

- **Overview** (umbrella) — the complete picture across the three pillars
- **AGORA** — metabolism → sign prior (input to the ecological model)
- **Dieckow** — in vivo interaction inference and validation (the present deck)
- **Network** — structural analysis of the interaction matrix $A$
- **Spatial-PDE** — reaction–diffusion of the FISH depth profiles
- **FISH pipeline** — .lif → 5-species depth composition

---

## Scope

From **compositional 16S time-series** of an oral biofilm, infer a **signed
interaction matrix** $A$, and quantify how far that inference can be trusted.

- model: generalized Lotka–Volterra (gLV) / Hamilton replicator on the simplex
- constraint: an AGORA2-derived **metabolic sign prior** (see the AGORA deck;
  cited here only briefly)
- focus: the **fit**, **LOO-CV**, model comparison, and **critical validation**

\vspace{0.4em}
**Claim.** The prior provides sign-consistency and interpretability rather than
predictive accuracy. What is validated independently of the data is the
**cross-feeding direction only** (below).

---

## Data and design

Dieckow et al. 2024 (ENA **PRJEB71108**): a longitudinal 16S series of
**10 patients $\times$ 3 weeks**, aggregated to **10 class-level guilds**
(`GUILD_ORDER` is canonical).

\begin{center}\includegraphics[width=0.86\textwidth,keepaspectratio]{dieckow_paper/figures/fig0_study_design.png}\end{center}

Early peri-implant colonization. Each sample is treated as a composition
$\varphi\in\Delta^{9}$ with $\sum_i\varphi_i=1$.

---

## Model — gLV and replicator

Generalized Lotka–Volterra for absolute abundances $x_i$:
$$\dot{x}_i = x_i\Big(b_i + \sum_{j=1}^{S} A_{ij}\,x_j\Big),\qquad S=10 .$$

On the simplex $\varphi_i=x_i/\sum_k x_k$, the replicator / Hamilton form:
$$\dot{\varphi}_i = \varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big],
\qquad \sum_i\varphi_i=1 .$$

- **Hamilton** uses a **symmetric** $A$ (replicator with linear payoffs;
  Taylor & Jonker 1978).
- **Classic gLV** uses an **asymmetric** $A$. $A_{ij}>0$: facilitation;
  $A_{ij}<0$: suppression.

---

## Inference is underdetermined $\Rightarrow$ sign prior

Parameters $\mathcal{O}(S^2)\approx100$ vs $10\times3$ observations. Abundance
alone cannot identify the signs of $A$ $\Rightarrow$ a one-sided hinge penalises
**sign violations only**:
$$\mathcal{L}(A,b)=\frac{1}{2\sigma^{2}}\sum_{t}\big\lVert\varphi^{\text{obs}}_{t}-\varphi^{\text{pred}}_{t}\big\rVert^{2}
\;+\; W\!\!\sum_{(i,j)\in\mathcal{M}}\!\!\relu{-\,P_{ij}\,A_{ij}} .$$

- $P_{ij}=\sgn(F_{ij})$ comes from AGORA cross-feeding flux (derived in the AGORA deck).
- $\relu{-P_{ij}A_{ij}}$ is $0$ when $\sgn(A_{ij})=P_{ij}$ — **magnitude is free**.
- Posterior sampled by **TMCMC** ($10^{4}$ particles).

---

## Fitted interaction matrix $A$

\begin{center}\includegraphics[height=0.62\textheight,keepaspectratio]{dieckow_paper/figures/fig1_A_matrix.png}\end{center}

Diagonal shows **self-limitation** $A_{ii}<0$ (resource limitation / saturation).
Off-diagonal: Actinobacteria $\leftrightarrow$ Bacilli **facilitation**.

---

## Trajectory fit (all 10 patients)

\begin{center}\includegraphics[height=0.68\textheight,keepaspectratio]{dieckow_paper/figures/fig_fitting_results.png}\end{center}

Observations (points) and predicted trajectories (lines). Per-patient $b$ vectors
with a shared matrix $A$.

---

## Sign correspondence: $A_{ij}$ vs metabolic $F_{ij}$

\begin{center}\includegraphics[width=0.92\textwidth,keepaspectratio]{dieckow_paper/figures/fig5b_aij_fij_scatter.png}\end{center}

$x$-axis = metabolic flow $F_{ij}$; $y$-axis = fitted $\hat A_{ij}$.
At $W=1.0$, $\sgn(\hat A_{ij})=\sgn(F_{ij})$ holds on the constrained set $\mathcal{M}$.

---

## LOO-CV definition

Re-fit with patient $p$ held out, evaluate on its trajectory:
$$\mathrm{RMSE}=\sqrt{\frac{1}{NT}\sum_{t}\big\lVert\varphi^{\text{obs}}_{t}-\varphi^{\text{pred}}_{t}\big\rVert^{2}},$$
$$\mathrm{BC}=\frac{\sum_i\lvert\varphi^{\text{obs}}_i-\varphi^{\text{pred}}_i\rvert}{\sum_i\big(\varphi^{\text{obs}}_i+\varphi^{\text{pred}}_i\big)}
\quad(\text{Bray–Curtis dissimilarity}).$$

Every one of the 10 patients is held out in turn (leave-one-patient-out) and averaged.

---

## LOO model comparison

\begin{center}\includegraphics[width=0.50\textwidth,keepaspectratio]{dieckow_paper/figures/fig2_loo_comparison.png}\hfill\includegraphics[width=0.48\textwidth,keepaspectratio]{dieckow_paper/figures/fig8_all_models_rmse_bc.png}\end{center}

\begin{center}\small LOO-RMSE / LOO-BC across prior layers, model forms, and prior stiffness $W$.\end{center}

---

## Performance summary

Best model **L1+L2+AGORA, $W=1.0$**:

| Metric | Value |
|---|---|
| train RMSE | $0.0565$ |
| Pearson $r$ | $0.951$ |
| sign agreement | $70/70$ ($100\%$) |
| **LOO-RMSE** | $\mathbf{0.0504}$ |
| LOO-BC | $0.1468$ |

- L1+L2 only: LOO-RMSE $0.0516$.
- **Prior-free gLV**: LOO-RMSE $0.0455$ (lower — more params, metabolically unconstrained).
- **MacArthur magnitude prior FAILED**: SA $4\text{–}8/70$.

\vspace{0.2em}
$\Rightarrow$ the prior provides **sign-consistency and interpretability** rather than raw accuracy.

---

## LOO stability of the $A$ matrix

\begin{center}\includegraphics[width=0.92\textwidth,keepaspectratio]{dieckow_paper/figures/fig7_loo_stability.png}\end{center}

Across the 10 hold-out re-fits, every pair has **sign-consistency $\geq 0.70$**.
The inferred sign structure is robust to leaving out any single patient.

---

## Critical validation

With the prior **OFF** ($\alpha=0$) the prior is **all-positive**, so naive
sign-agreement is inflated. A **label permutation test** ($n=10^{4}$) controls for this:

| Model | cross-feeding direction | competition |
|---|---|---|
| **Hamilton $\alpha=0$** | $\mathbf{78.6\%}$ (11/14) vs random $37.7\%$, $p=4\times10^{-4}$, $z=+3.79$ | $\approx$ chance |
| gLV | $41\%$ (null) | null |

\begin{center}\includegraphics[height=0.34\textheight,keepaspectratio]{results/dieckow_cr/loo_alpha_comparison_hamilton_noagora_glv_noagora.png}\end{center}

**Only the cross-feeding direction is validated. Competition is not.**

---

## Two-cohort replication

Botelho 2021 (**PRJNA725874**, 15 patients $\times$ 7 timepoints) is fit
**prior-free** and independently, then strong-pair signs are compared to Dieckow:

\begin{center}\includegraphics[width=0.94\textwidth,keepaspectratio]{results/botelho_validation/fig_botelho_A_comparison_noprior.png}\end{center}

Strong-pair directed signs agree at **$89\%$** ($8/9$ upper-triangular,
$p\approx0.02$). The **Actinobacteria axis** is consistent across cohorts.

---

## Interpretation and limitations

- The **AGORA prior itself is not reproduced** by the 16S dynamics $\Rightarrow$
  it is a **modelling choice**, not a data-confirmed fact.
- What is validated independently of the data:
  1. the **cross-feeding direction** (Hamilton, $p=4\times10^{-4}$),
  2. **two-cohort strong-pair** signs (Dieckow $\times$ Botelho, $89\%$).
- Competition direction, prior magnitude, and the full matrix are not validated.
- The MICOM–gLV RMSE improvement ($p\approx0.07$, $N=10$) is **structurally underpowered** (power $\approx0.30$ for $d=0.5$); read as a consistency result, not a powered effect. The permutation test ($p=4\times10^{-4}$) is the primary statistical claim.
- A **neutral-initialised no-prior LOO** (zero-init, submitted) will confirm the enrichment is data-driven and not a warm-start artefact.

\vspace{0.3em}
To avoid overstating the result: a **subset** of signs is supported; the remainder remains hypothesis.

---

## External clinical validation (in progress)

Plan to compare a Guild Dysbiosis Index against the Joshi 2025 cohort
(**PRJNA1192962**):

- the index is derived from the inferred attractor of the fitted $A$ matrix,
  then scored against the cohort's health/disease labels;
- this provides an external, patient-level test that is independent of the
  Dieckow training data.

\vspace{0.4em}
\textcolor{red}{\textbf{Preliminary}: awaiting clinical metadata. The
health/disease matching is not yet complete; results will be added once the
cohort metadata is released.}

---

## Conclusions

1. Compositional 16S time-series $\to$ a signed $A$, via gLV / Hamilton replicator.
2. Under-determination is eased by the **AGORA metabolic sign prior** (one-sided
   hinge); posterior via TMCMC.
3. Best: L1+L2+AGORA $W=1.0$ gives **LOO-RMSE $0.0504$**, SA $100\%$. But
   prior-free gLV reaches $0.0455$ — the prior provides **interpretability** rather than accuracy.
4. **Signs are stable under LOO** ($\geq0.70$); the MacArthur magnitude prior fails.
5. Honest validation: only the **cross-feeding direction** ($p=4\times10^{-4}$) and
   **two-cohort strong pairs** ($89\%$) are independently supported. Clinical
   validation is in progress.
