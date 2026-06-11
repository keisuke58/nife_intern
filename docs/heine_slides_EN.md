---
title: "GPU-Accelerated Bayesian Inference of Multi-Species Biofilm Interaction Parameters via TMCMC"
subtitle: "Full discovery of the 15-dimensional interaction matrix $A$ with two-phase GPU-TMCMC"
author: "Keisuke Nishioka, Felix Klempt, Hendrik Geisler, Meisam Soleimani, Philipp Junker (IKM, LUH); R. Mukherjee, S. P. Szafrański, M. Stiesch (MHH / NIFE)"
date: "2026-06-11"
theme: "Madrid"
colortheme: "whale"
aspectratio: 169
classoption:
  - "c"
header-includes:
  - \usepackage{amsmath,amssymb}
  - \newcommand{\sgn}{\operatorname{sgn}}
---

## Motivation — peri-implantitis and biofilm dysbiosis

Peri-implantitis is driven by a shift of the peri-implant oral biofilm from a
**commensal** to a **dysbiotic** multi-species community.

- The ecological driver is the **species–species interaction matrix** $A$.
- **Goal:** infer $A$ from compositional time-series, and quantify how the
  health $\to$ disease transition reshapes it.
- **Thesis.** All 15 interaction parameters are recovered from data alone
  (no a-priori sign locking); the health/disease difference is a
  **topological reorganisation**, not a simple rescaling.

---

## Data — Heine et al. (2025), four conditions

Longitudinal relative-abundance time-series of a **five-species** oral biofilm,
under **four conditions**:

- **Commensal** vs **dysbiotic** community $\times$ **static** vs **HOBIC**
  reactor flow $\Rightarrow$ CS, CH, DS, DH.
- Five species: *S. oralis* (So), *A. naeslundii* (An), *Veillonella* (Vei),
  *F. nucleatum* (Fn), *P. gingivalis* (Pg).
- Each sample is a composition $\varphi\in\Delta^{4}$ with $\sum_i\varphi_i=1$.

---

## Model — continuum biofilm / replicator dynamics

State: relative abundances $\varphi_i$ on the simplex. The Extended Hamilton
Principle yields replicator dynamics with a **symmetric** interaction matrix $A$:
$$\dot{\varphi}_i = \varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big],
\qquad \sum_i\varphi_i = 1 .$$

- Symmetric $A$ (5$\times$5) $\Rightarrow$ **15 free parameters**.
- $A_{ij}>0$: facilitation; $A_{ij}<0$: suppression / competition.

---

## Inference — Bayesian framework

$$\mathcal{L}(A,b)=\frac{1}{2\sigma^{2}}\sum_{t}\big\lVert\varphi^{\text{obs}}_{t}-\varphi^{\text{pred}}_{t}\big\rVert^{2}
\;+\; \text{(weakly-informative prior)} .$$

- **All 15 entries free** — no a-priori sign constraint, so any sparsity is
  data-driven.
- Posterior sampled by **Transitional MCMC (TMCMC)** with tempered likelihood
  and $N_p$ particles.

---

## Two-phase estimation strategy

- **Phase 1** (fix $\psi$, $N_p=1{,}000$): rapid, robust identification of $A$.
- **Phase 2** (free $\psi$, $N_p=10{,}000$): full joint posterior, physically
  consistent.

Phase-1 MAP is already close to Phase-2 ($\Delta\mathrm{RMSE}\le 0.03$),
confirming the fix-$\psi$ stage is a robust warm start.

---

## GPU acceleration

- **JAX + `vmap`** on a single RTX 4090.
- **≈200× speedup** vs CPU; $N_p=10{,}000$ in ≈2 h per condition.
- Makes production runs of the full **15-D posterior** feasible.

---

## Posterior predictive fits — all four conditions

\begin{center}\includegraphics[height=0.80\textheight,keepaspectratio]{docs/paper_figs/hamilton_fit_thesis.pdf}\end{center}

Hamilton MAP $\varphi_i$ with 50/80\% CI vs experimental median $\pm$ replicate
IQR (5 species $\times$ 4 conditions). RMSE $0.033$–$0.119$; DS best-identified.

---

## Reference — classic gLV fits the data more tightly

\begin{center}\includegraphics[height=0.80\textheight,keepaspectratio]{results/heine2025/glv_heine_fit_thesis.pdf}\end{center}

Five species $\times$ four conditions: experimental median $\pm$ replicate IQR vs
gLV MAP. RMSE $0.012$–$0.032$. The asymmetric gLV fits more tightly, but the
Hamilton replicator trades a little fit for an **interpretable symmetric** $A$.

---

## Inferred interaction matrices — commensal vs dysbiotic

\begin{center}\includegraphics[width=0.92\textwidth,keepaspectratio]{results/heine_repro/heatmap_A_4cond.pdf}\end{center}

- **Commensal (CS, CH):** So–An and So–Vei blocks dominant; Fn, Pg entries
  $\approx 0$ — **emergent sparsity** without locking.
- **Dysbiotic (DS, DH):** strong **Vei$\to$Pg** (pH) and **Fn$\to$Pg**
  (peptides); $\hat a^{\,DH}_{45}=5.63$.

---

## Interaction matrices across conditions — UMAP embedding

\begin{center}\includegraphics[height=0.66\textheight,keepaspectratio]{docs/paper_figs/umap_A_thesis.pdf}\end{center}

The four posterior $A$-matrices separate in interaction space: commensal
(CS, CH) and dysbiotic (DS, DH) form distinct clusters — the health/disease
shift is a structural reorganisation, not a small perturbation.

---

## Posterior parameter distributions (Phase 2, $N_p=10{,}000$)

\begin{center}\includegraphics[width=0.92\textwidth,keepaspectratio]{docs/paper_figs/posterior_violin_thesis.pdf}\end{center}

Pg-related parameters ($a_{15},a_{25},a_{35},a_{45},a_{55}$) concentrate near
zero in CS/CH **without a-priori locking**, and activate in DS/DH.

---

## Quantitative fit metrics and phase consistency

| Cond. | Phase 1 RMSE | Phase 2 RMSE | Phase 2 MAE | Phase MAP $r$ |
|---|---|---|---|---|
| CS | $0.119$ | $0.119$ | $0.075$ | $0.87$ |
| CH | $0.071$ | $0.104$ | $0.088$ | $0.68$ |
| DS | $0.020$ | $0.033$ | $0.024$ | $0.42$ |
| DH | $0.067$ | $0.087$ | $0.060$ | $0.62$ |

- DS: lowest RMSE $=0.033$ (best identifiability).
- Phase-2 $\Delta\mathrm{RMSE}\le 0.03$: Phase 1 is robust.

---

## Two-phase consistency — Phase 1 vs Phase 2 MAP

\begin{center}\includegraphics[width=0.95\textwidth,keepaspectratio]{docs/paper_figs/phase1_vs_phase2.pdf}\end{center}

Phase-1 (fix-$\psi$) and Phase-2 (free $\psi$) MAP estimates of $A$ agree closely
(Pearson $r$ $0.42$–$0.87$), confirming the fast fix-$\psi$ stage as a robust
warm start for the full posterior.

---

## Independent validation — pH prediction

\begin{center}\includegraphics[width=0.92\textwidth,keepaspectratio]{docs/paper_figs/ph_validation_thesis.pdf}\end{center}

pH predicted from the posterior $A$ via the Hamilton ODE — **not used in
calibration**. Independent $R^2=0.78$ (RMSE $0.13$, $N=12$ held-out;
LOO $R^2=0.92$); lactate-driven acidification. Gingipain–Pg: $r=0.90$.

---

## Discussion — biological interpretation

**Commensal state (CS, CH)**

- So–An co-aggregation + So–Vei lactate exchange: a stable **mutualistic core**.
- Fn, Pg entries emerge near zero — data-driven sparsity.

**Dysbiotic transition (DS, DH)**

- **Vei$\to$Pg** (pH) and **Fn$\to$Pg** (peptides) strongly activated by W83.
- $\rho(\mathrm{DH}-\mathrm{CS})=-0.49$: **topological reorganisation**, not
  recoverable by parameter rescaling.

---

## Conclusions

1. **Two-phase full-discovery TMCMC:** Phase 1 (fix-$\psi$) gives rapid robust
   $A$; Phase 2 (free $\psi$) gives a consistent joint posterior. All 15
   parameters free — **emergent sparsity** from data.
2. **GPU-accelerated TMCMC:** ≈200× speedup; $N_p=10{,}000$ in ≈2 h per
   condition.
3. **Validated interpretation:** pH $r=0.84$, gingipain $r=0.90$; dysbiosis is a
   **topological reorganisation** ($\rho=-0.49$).

---

## Future work

- **6-species model** (Siddiqui et al. 2021, cpTi, 21 days).
- **Mechanistic pH** (Henderson–Hasselbalch + lactate from So/Vei).
- Extension to additional longitudinal cohorts for cross-study validation.

\vspace{0.4em}
Code and data available upon request.

---

## Appendix A — FISH (.lif): depth-resolved composition

\begin{center}\includegraphics[width=0.92\textwidth,keepaspectratio]{results/fig_fish_zprofile_temporal.pdf}\end{center}

Depth-resolved relative abundance of the five species (CH top, DH bottom) over
21 days, from confocal FISH (.lif) z-stacks. In the dysbiotic condition Pg
accumulates with depth and time — the spatial dimension that motivates future
Hamilton + reaction–diffusion modelling.

---

## Appendix B — raw .lif 3-D structure (5 channels)

\begin{center}\includegraphics[width=0.95\textwidth,keepaspectratio]{results/fish_3d/fish3d_proj_DH_d6_s0.png}\end{center}

Confocal FISH (.lif) of the dysbiotic biofilm, day 6: per-species fluorescence,
lateral $xy$ projection (top) and depth $xz$ section (bottom). Pg occupies the
deeper layers, consistent with the depth profiles.

---

## Appendix C — raw .lif: Pg orthogonal views

\begin{center}\includegraphics[width=0.95\textwidth,keepaspectratio]{results/fish_3d/fish3d_ortho_P.g_DH_d6_s0.png}\end{center}

Orthogonal ($xy$/$xz$/$yz$) sections of *P. gingivalis* in the dysbiotic biofilm
— a spatially structured, depth-biased distribution rather than a uniform layer.

---

## Appendix D — FISH Fn–Pg co-localization

\begin{center}\includegraphics[width=0.88\textwidth,keepaspectratio]{results/fish_3d/fish_3d_fnpg_coloc.pdf}\end{center}

Manders $M_1$ (Pg with Fn) over time, lateral patch ($xy$) and 3-D voxels
(84 FOV). Fn–Pg remain spatially associated, consistent with the inferred
**Fn$\to$Pg** interaction.

---

## Appendix E — FISH lateral heterogeneity

\begin{center}\includegraphics[width=0.92\textwidth,keepaspectratio]{results/fish_3d/fish_3d_lateral_heterogeneity.png}\end{center}

Lateral (field-of-view to field-of-view) variability of the depth-resolved
composition — quantifying the spatial heterogeneity not captured by the
well-mixed ODE.

---

## Appendix F — FISH vs well-mixed model

\begin{center}\includegraphics[width=0.86\textwidth,keepaspectratio]{results/fig_fish_vs_experiment.png}\end{center}

Depth-averaged FISH composition vs the well-mixed ODE: broad agreement on the
dominant species, while FISH additionally resolves the spatial structure the
ODE cannot represent.
