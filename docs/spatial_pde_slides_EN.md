---
title: "Spatial Reaction–Diffusion of the Biofilm (Heine HOBIC)"
subtitle: "Inverse estimation of spatial transport from FISH depth profiles — formal treatment"
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
- **AGORA** — metabolism → sign prior (input to the ecological model)
- **Dieckow** — in-vivo interaction inference & validation (the model)
- **Network** — structural analysis of the interaction matrix $A$
- **Spatial-PDE** — reaction-diffusion of the FISH depth profiles  — **(you are here)**
- **FISH pipeline** — .lif → 5-species depth composition

---

## Question: adding WHERE to WHO

The bulk ODE fit gives **who interacts with whom (WHO)**; the FISH depth
profiles add **where that interaction happens (WHERE)**.

- Known: bulk gLV/Hamilton interaction matrix $A$ and growth vector $b$
  (Dieckow fit).
- Unknown: per-species **transport** along the substratum depth axis.
- Goal: with the reaction term fixed from the bulk fit, infer **only the
  spatial transport parameters** from the depth-resolved FISH data.

\vspace{0.4em}
**Thesis.** Dysbiosis manifests not as a change in bulk composition but as a
**spatial reorganisation** — in particular *P. gingivalis* sinks deep and
decouples from *F. nucleatum*.

---

## Data: HOBIC flow-chamber FISH (Heine 2025)

**11 `.lif` files**, two conditions $\times$ five days, titanium substratum:

- Conditions: **CH** (commensal, HOBIC22) and **DH** (dysbiotic, HOBIC24).
- Days: 1 / 6 / 10 / 15 / 21.
- Five species: So (*S. oralis*), An (*Actinomyces*), Vd/Vp (*Veillonella*),
  Fn (*F. nucleatum*), Pg (*P. gingivalis*).
- Pooled FOV counts: CH $=15/5/15/16/16$, DH $=7/3/3/2/2$.

\vspace{0.3em}
The 4-channel FISH is decoded into 5 species (*F. nucleatum* $=$ blue $\cap$ red
double label; see FISH deck). The $z$-axis voxel intensities give the
depth-resolved composition $\varphi_i(z,t)$.

---

## Depth profiles: CH vs DH

![](results/diffusion_fit/zprofiles_all_ti_overlay.png){ height=68% }

Per-species $\varphi_i(z)$ overlaid for CH (commensal) and DH (dysbiotic).
Depth $z$ runs from the substratum ($z=0$) to the bulk ($z=L$).

---

## Per-depth stacked composition

![](results/diffusion_fit/zprofiles_all_ti_stacked.png){ height=68% }

Stacked $\{\varphi_i(z)\}_i$ at each depth $z$. At every $z$ the constraint
$\sum_i \varphi_i + \varphi_0 = 1$ holds ($\varphi_0$ = void / unoccupied).

---

## Governing equation (reaction–diffusion PDE)

For depth $z\in[0,L]$ and composition $\varphi_i(z,t)$, the
reaction–advection–diffusion equation is
$$\frac{\partial \varphi_i}{\partial t}
 = D_i\,\frac{\partial^2 \varphi_i}{\partial z^2}
 \;-\; u\,\frac{\partial \varphi_i}{\partial z}
 \;+\; R_i(\varphi).$$

- $R_i(\varphi)$ is the replicator / Hamilton reaction; $A,b$ are **fixed** from
  the bulk fit,
  $$R_i(\varphi)=\varphi_i\Big[(A\varphi+b)_i - \varphi^{\!\top}(A\varphi+b)\Big]+\gamma\,\varphi_i.$$
- The **only free parameters are the diffusivities $D_i$ and the advection $u$**.
- $\gamma$ is a Lagrange term enforcing $\sum_i \varphi_i + \varphi_0 = 1$ at each $z$.

---

## Boundary conditions and numerics

**Boundary conditions:**
$$\left.\frac{\partial \varphi_i}{\partial z}\right|_{z=0}=0\quad(\text{substratum, no-flux}),\qquad
\varphi_i\big|_{z=L}=\varphi_{\text{bulk},i}\quad(\text{bulk, Dirichlet}).$$
$\varphi_{\text{bulk},i}$ is fixed to the Day-1 median.

**Numerics (Method of Lines):** Lie operator splitting alternates reaction and
transport,
$$e^{\Delta t\,\mathcal{L}} \approx e^{\Delta t\,\mathcal{R}}\,e^{\Delta t\,\mathcal{T}} + \mathcal{O}(\Delta t).$$

- Reaction $\mathcal{R}$: implicit Euler with Newton iteration per $z$-node (stiff).
- Transport $\mathcal{T}$: explicit finite difference (central $\partial_z^2$, upwind $\partial_z$).

---

## Inverse problem

Least-squares estimation of the transport parameters against the observation
$\varphi^{\text{obs}}(z,t)$:
$$\min_{D,\,u}\;\; \sum_t \sum_z
 \big\lVert \varphi^{\text{pred}}(z,t;D,u) - \varphi^{\text{obs}}(z,t)\big\rVert^2 .$$

- Optimiser: **L-BFGS-B**, with a numerical gradient through the stiff PDE
  (JAX forward model).
- The forward model is the MoL solver above; with $A,b$ fixed the unknowns are
  $D\in\mathbb{R}^5$ and $u$.
- Non-dimensionalisation: the **Péclet number** $\mathrm{Pe} = uL/D_i$ measures
  advection vs diffusion ($\mathrm{Pe}\gg1$ advection-dominated,
  $\mathrm{Pe}\ll1$ diffusion-dominated).

---

## Predicted vs observed depth profiles

::: columns
:::: column
![](results/diffusion_fit/fit_CH.png){ height=62% }

CH (commensal)
::::
:::: column
![](results/diffusion_fit/fit_DH.png){ height=62% }

DH (dysbiotic)
::::
:::

Predicted (solid) vs observed (points) depth profiles. Reaction term fixed,
transport only fitted.

---

## Fitted parameters (preliminary)

| Species | $D$ (CH) | $D$ (DH) |
|---|:---:|:---:|
| So | $0.0123$ | $0.0596$ |
| An | $0.0121$ | $0.0021$ |
| Vd/Vp | $0.0042$ | $2.1\times10^{-5}$ |
| Fn | $0.0075$ | $0.0019$ |
| Pg | $0.0115$ | $0.0083$ |
| **$u$ (advection)** | $0.0038$ | $0.0060$ |
| **loss** | $0.128$ | $0.102$ |

\vspace{0.2em}
\textcolor{red}{**Preliminary**: both conditions report `success=False` (optimiser not converged).}
A hyperparameter sweep with faster-converging settings is running on the HPC.

---

## Spatial-ecology finding 1: Pg sinks deep in dysbiosis

![](results/diffusion_fit/depth_niche.png){ height=58% }

The center-of-mass of *P. gingivalis* shifts up to $+30\,\mu\mathrm{m}$ **deeper**
in DH from Day 6 onward — sedimentation into the deep anaerobic niche.

---

## Spatial-ecology finding 2: Fn–Pg bridging is early-only

![](results/diffusion_fit/fn_pg_coloc.png){ height=50% }

Manders $M_1$ (fraction of Pg co-localised with Fn):
DH Day 1 $=0.76$, then $0.15\text{–}0.20$ from Day 6; CH stays $0.33\text{–}0.49$.
$\Rightarrow$ Pg **decouples** from Fn and goes autonomous.

---

## Other: bulk CH ≈ DH

The CH–DH **Bray–Curtis divergence** is $\approx 0.2$ and **flat from Day 1**.

- The difference is set by the inoculum and barely grows over time.
- In bulk composition CH $\approx$ DH $\Rightarrow$ dysbiosis is **spatial /
  temporal**, not a shift in bulk composition.

\vspace{0.4em}
This is consistent with findings 1–2: the difference is not "who and how much"
but "where they sit and what they partner with".

---

## Diffusivity vs ecological centrality

![](results/diffusion_fit/d_vs_centrality.png){ height=52% }

The fitted diffusivity $D_i$ correlates **negatively** with gLV interaction
strength (centrality): CH $r=-0.40$, DH $r=-0.47$ (5 species, illustrative).

\vspace{0.2em}
$\Rightarrow$ "central species move less": ecological hubs tend to be spatially
pinned.

---

## In-vitro ↔ in-vivo mapping

![](results/diffusion_fit/hobic_vs_dieckow.png){ height=50% }

Mapping the 5 HOBIC species onto the 5 Dieckow guilds, the **rank order agrees**:
Spearman $\rho = 0.70$.

- In-vivo is *S. oralis*-dominated; the defined inoculum is even.
- The rank structure is preserved $\Rightarrow$ the in-vitro model is a valid
  reduction of the in-vivo community.

---

## Conclusions

1. We build a framework that adds FISH depth profiles (WHERE) to the bulk ODE
   (WHO) and, with the reaction term fixed, infers **only the transport
   parameters $D_i,u$**.
2. **Dysbiosis $=$ spatial reorganisation**: *P. gingivalis* sinks deep
   ($+30\,\mu\mathrm{m}$) and goes autonomous from *F. nucleatum* (Manders $M_1$
   high only early).
3. Bulk is CH $\approx$ DH (Bray–Curtis $\approx0.2$, flat) — the difference is
   spatial, not bulk.
4. "Central species move less" ($D$ anti-correlates with centrality); the
   in-vitro↔in-vivo rank order agrees ($\rho=0.70$).
5. \textcolor{red}{Transport parameters are **preliminary** (not converged)};
   a sweep with faster settings is converging on the HPC.
