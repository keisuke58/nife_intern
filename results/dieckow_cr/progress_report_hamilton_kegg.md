---
marp: true
theme: default
paginate: true
math: mathjax
style: |
  :root {
    --navy:  #1a3a5c;
    --blue:  #4C8DC4;
    --green: #3a9e5f;
    --red:   #c0392b;
    --bg:    #f7f9fc;
  }
  section {
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 21px;
    background: white;
    color: #222;
    padding: 48px 56px;
  }
  section.lead {
    background: var(--navy);
    color: white;
    text-align: center;
  }
  section.lead h1 { color: white; font-size: 44px; margin-bottom: 8px; }
  section.lead h2 { color: #a8c8e8; font-size: 24px; border: none; font-weight: 400; }
  section.lead p  { color: #cde; font-size: 18px; margin-top: 32px; }
  h1 { font-size: 34px; color: var(--navy); margin-bottom: 12px; }
  h2 { font-size: 26px; color: var(--navy);
       border-bottom: 2.5px solid var(--blue);
       padding-bottom: 5px; margin-bottom: 18px; }
  h3 { font-size: 20px; color: var(--blue); margin: 10px 0 4px; }
  a  { color: var(--blue); }

  /* callout boxes */
  .box {
    background: #eef4fb;
    border-left: 4px solid var(--blue);
    border-radius: 4px;
    padding: 10px 16px;
    margin: 12px 0;
    font-size: 20px;
  }
  .box-green {
    background: #edfaf3;
    border-left: 4px solid var(--green);
    border-radius: 4px;
    padding: 10px 16px;
    margin: 12px 0;
    font-size: 20px;
  }
  .box-red {
    background: #fdf0ef;
    border-left: 4px solid var(--red);
    border-radius: 4px;
    padding: 10px 16px;
    margin: 12px 0;
  }

  /* two-column layout */
  .cols { display: flex; gap: 32px; align-items: flex-start; }
  .col  { flex: 1; }

  /* metric cards */
  .metrics { display: flex; gap: 20px; margin: 18px 0; }
  .card {
    flex: 1; text-align: center;
    background: var(--bg);
    border: 1.5px solid #c8daf0;
    border-radius: 8px;
    padding: 12px 8px;
  }
  .card .val { font-size: 32px; font-weight: 700; color: var(--navy); }
  .card .lbl { font-size: 14px; color: #666; margin-top: 2px; }

  table { font-size: 19px; border-collapse: collapse; width: 100%; margin-top: 8px; }
  th { background: var(--navy); color: white; padding: 8px 14px; }
  td { padding: 7px 14px; border-bottom: 1px solid #ddd; }
  tr:last-child td { border-bottom: none; }
  tr:nth-child(even) td { background: var(--bg); }

  footer { font-size: 14px; color: #aaa; }
  section::after { color: #aaa; font-size: 14px; }

  img[alt~="center"] { display: block; margin: 0 auto; }
---

<!-- _class: lead -->

# Hamilton + KEGG-prior Guild Model

## Progress Report — Oral Microbiome Community Dynamics

Keisuke Nishioka &nbsp;·&nbsp; 2026-05-04

---

## Background & Motivation

**Research question**: Can metabolomic network constraints improve community dynamics modelling?

<div class="cols">
<div class="col">

### Existing models
| Model | LOO-CV RMSE |
|---|---|
| gLV (free) | 0.0588 |
| gLV + KEGG | 0.0741 |
| Hamilton (free) | 0.0855 |
| **Hamilton + KEGG** | ← **this work** |

</div>
<div class="col">

### Approach
- **10-guild** replicator (class-level taxonomy)
- **8 patients**, 3 time points (weeks 1→2→3)
- Sign prior derived from **KEGG/HMDB** metabolite flow matrix $F$
- Fit via JAX autodiff + L-BFGS-B on RTX 3090

</div>
</div>

<div class="box">

**Hypothesis**: KEGG sign constraints regularize $A$ towards biologically meaningful interactions without sacrificing fit quality.

</div>

---

## Method

**Hamilton replicator ODE**

$$\dot{\phi}_i = \phi_i \!\left( b_i + \sum_j A_{ij}\,\phi_j - \bar{f}(\phi) \right), \qquad \bar{f} = \sum_i \phi_i f_i$$

**Loss function with metabolite sign penalty**

$$\mathcal{L} = \underbrace{\text{RMSE}(\hat\phi, \phi)}_{\text{data fit}} \;+\; \underbrace{\sum_{(i,j):\,F_{ij}\neq 0} \frac{|F_{ij}|}{2\sigma^2}\,\max\!\bigl(0,\,-\operatorname{sgn}(F_{ij})\cdot A_{ij}\bigr)^2}_{\text{metabolite sign penalty }(\sigma=0.15)} \;+\; \lambda\|A\|^2$$

<div class="cols" style="margin-top:16px">
<div class="col">

**Sign prior — data source**
- **Szafranski et al. Suppl. File 1** (literature-curated microbe × metabolite interactions: PRODUCES / USES / IS_INHIBITED_BY)
- KEGG compound ID / HMDB ID → **confidence weight only** (2.0 if annotated, 1.0 otherwise); no live API query
- $F[i,j]>0$: guild $j$ produces what $i$ uses &nbsp;→&nbsp; expect $A_{ij}>0$
- $F[i,j]<0$: guild $j$ produces inhibitor of $i$ &nbsp;→&nbsp; expect $A_{ij}<0$

</div>
<div class="col">

**Optimization**
- Warm start from Hamilton free fit
- JAX `jax.vmap` over patients
- scipy L-BFGS-B + analytical gradients
- Diagonal $A_{ii} \leq 0$ (self-limitation)
- 22 sign-constrained pairs (after symmetrisation)

</div>
</div>

---

## Result 1 — Interaction Matrix $A$

![center w:950](fig_guild_hamilton_kegg_Amatrix.png)

<div class="box-green">

**Sign agreement = 22 / 22 (100%)** with KEGG metabolite network &nbsp;·&nbsp; Green border ✓ = supported &nbsp;·&nbsp; Red ✗ = discordant

</div>

---

## Result 2 — Trajectory Prediction (Week 1 → 2 → 3)

![center w:940](fig_slide_trajectory.png)

<div class="metrics" style="margin-top:10px">
  <div class="card"><div class="val">0.874</div><div class="lbl">Pearson r (overall)</div></div>
  <div class="card"><div class="val">0.0830</div><div class="lbl">RMSE (overall)</div></div>
  <div class="card"><div class="val">0.987</div><div class="lbl">r — best patient (G)</div></div>
  <div class="card"><div class="val">0.148</div><div class="lbl">r — worst patient (K)</div></div>
</div>

---

## Result 3 — Model Comparison (LOO-CV)

![center w:840](fig_guild_model_comparison.png)

<div class="box-green">

**Hamilton + KEGG LOO-CV = 0.0770** &nbsp;·&nbsp; 9.4% improvement over Hamilton free (0.0855) &nbsp;·&nbsp; Competitive with gLV + KEGG (0.0741, Δ = 3.9%)

</div>

---

## Key Findings

<div class="cols">
<div class="col">

### What works ✓
- **Full sign consistency**: 22/22 (100%) KEGG agreement — metabolomic constraints are compatible with Hamilton dynamics
- **KEGG prior regularises effectively**: LOO-CV 0.0855 → 0.0770 (9.4% improvement)
- **Competitive generalisation**: within 3.9% of gLV + KEGG at 0.0741
- **Biologically interpretable** $A$: negative self-regulation, positive Actinobacteria ↔ Bacilli mutualism
- **Steady-state check**: RMSE 0.177 (vs ODE 0.083) — data are transient, not equilibrium

</div>
<div class="col">

### Remaining gap vs gLV
- gLV free still best at **0.0588** — Hamilton's transient approximation (short $t$) limits fit quality
- Patient A outlier (LOO-RMSE 0.159) drives variance; community type differs from training set
- **Next**: expanded flow (34 pairs, all 10 patients) fit running → may close the gap

### Open questions
- Do **34 KEGG+AGORA pairs** further improve LOO-CV?
- Optimal $\sigma$ is **insensitive** ($\sigma \in [0.05, 0.30]$ all give SA = 22/22, RMSE ≈ 0.081)

</div>
</div>

---

## Next Steps

| # | Task | Status |
|---|---|---|
| 1 | Hamilton + KEGG **LOO-CV** (8-fold) | ✅ **Done** — 0.0770 |
| 2 | **Aggregate** fold results & regenerate Fig 3 | ✅ Done |
| 3 | **Sign probability** heatmap from 10,000p posterior | ✅ Done |
| 4 | **Paper S1** update with correct posterior statistics | ✅ Done |
| 5 | **Expanded flow** (34 pairs, full 10 patients) fit | 🔄 Running GPU |
| 6 | **σ sensitivity** analysis | ✅ Insensitive — σ = 0.15 optimal |
| 7 | **Steady-state** reformulation | ✅ Tested — transient data, ODE preferred |
| 8 | **Network visualisation** of interaction matrix | ✅ Done |

<div class="box-green" style="margin-top: 20px">

**Summary**: Hamilton + KEGG prior achieves 100% metabolomic sign consistency (22/22) and 9.4% LOO-CV improvement over Hamilton free (0.077 vs 0.086). Generalisation is competitive with gLV + KEGG (Δ = 3.9%), while offering mechanistic interpretability that gLV lacks.

</div>
