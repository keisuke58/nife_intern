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
    font-size: 20px;
    background: white;
    color: #222;
    padding: 44px 52px;
  }
  section.lead {
    background: var(--navy);
    color: white;
    text-align: center;
  }
  section.lead h1 { color: white; font-size: 40px; margin-bottom: 8px; }
  section.lead h2 { color: #a8c8e8; font-size: 22px; border: none; font-weight: 400; }
  section.lead p  { color: #cde; font-size: 17px; margin-top: 28px; }
  h1 { font-size: 32px; color: var(--navy); margin-bottom: 10px; }
  h2 { font-size: 24px; color: var(--navy);
       border-bottom: 2.5px solid var(--blue);
       padding-bottom: 5px; margin-bottom: 16px; }
  h3 { font-size: 18px; color: var(--blue); margin: 8px 0 3px; }
  a  { color: var(--blue); }

  .box {
    background: #eef4fb;
    border-left: 4px solid var(--blue);
    border-radius: 4px;
    padding: 9px 14px;
    margin: 10px 0;
    font-size: 19px;
  }
  .box-green {
    background: #edfaf3;
    border-left: 4px solid var(--green);
    border-radius: 4px;
    padding: 9px 14px;
    margin: 10px 0;
    font-size: 19px;
  }
  .box-red {
    background: #fdf0ef;
    border-left: 4px solid var(--red);
    border-radius: 4px;
    padding: 9px 14px;
    margin: 10px 0;
  }
  .cite {
    font-size: 14px;
    color: #888;
    margin-top: 4px;
  }

  .cols { display: flex; gap: 28px; align-items: flex-start; }
  .col  { flex: 1; }

  .metrics { display: flex; gap: 16px; margin: 14px 0; }
  .card {
    flex: 1; text-align: center;
    background: var(--bg);
    border: 1.5px solid #c8daf0;
    border-radius: 8px;
    padding: 10px 6px;
  }
  .card .val { font-size: 28px; font-weight: 700; color: var(--navy); }
  .card .lbl { font-size: 13px; color: #666; margin-top: 2px; }

  table { font-size: 17px; border-collapse: collapse; width: 100%; margin-top: 8px; }
  th { background: var(--navy); color: white; padding: 7px 12px; }
  td { padding: 6px 12px; border-bottom: 1px solid #ddd; }
  tr:last-child td { border-bottom: none; }
  tr:nth-child(even) td { background: var(--bg); }

  footer { font-size: 13px; color: #aaa; }
  section::after { color: #aaa; font-size: 13px; }

  img[alt~="center"] { display: block; margin: 0 auto; }
---

<!-- _class: lead -->

# Guild-level Replicator Dynamics with Metabolite Sign Prior

## Progress Report — Oral Microbiome Community Dynamics

Keisuke Nishioka &nbsp;·&nbsp; NIFE · IKM, Leibniz University Hannover &nbsp;·&nbsp; 2026-05-05

---

## LOO-CV — What We're Testing

<div class="cols">
<div class="col">

**Setup** (10-fold, one patient out)

```
All 10 patients
─────────────────────────────
Fold 1:  [A] B C D E F G H K L
          ↑ held-out
Fold 2:  A [B] C D E F G H K L
              ↑ held-out
  ⋮
Fold 10: A B C D E F G H K [L]
                             ↑ held-out
```

For each fold:
1. Fit **A** on 9 patients
2. Fit only **b_p** for held-out patient (wk 1 data)
3. Predict wk 2 & 3 → measure error

</div>
<div class="col">

**Why it matters**

| | Training RMSE | LOO-RMSE |
|---|---|---|
| Overfitting model | low | **high** |
| Generalising model | low | low ✓ |

**Sign prior as regulariser** — constraining $A$ to biologically consistent signs reduces overfitting → smaller train/LOO gap

$$\text{LOO-RMSE} = \frac{1}{N}\!\sum_{p}\!\sqrt{\frac{\text{MSE}^{(p)}_{\text{wk2}} + \text{MSE}^{(p)}_{\text{wk3}}}{2}}$$

<div class="box-green">

**Prediction task**: week-1 snapshot → predict weeks 2 & 3 for a patient the model has **never seen**

</div>

</div>
</div>

---

## Background & Motivation

**Dataset**: Dieckow et al. (2024, *npj Biofilms Microbiomes*, DOI: 10.1038/s41522-024-00624-3)
&nbsp;&nbsp;— 12 patients, dental implant abutment biofilm, 16S rRNA amplicon sequencing, weeks 1–3

**Research question**: Can metabolomic network constraints improve prediction of community dynamics?

<div class="cols">
<div class="col">

### Model benchmark (LOO-CV RMSE)
| Model | Train | LOO | SA |
|---|---|---|---|
| gLV (free) | — | 0.0588 ★ | — |
| gLV + metabolite prior | — | 0.0741 | — |
| Hamilton free | — | 0.0855 | — |
| Replicator + prior (22 pairs) | — | 0.0770 | 22/22 |
| Replicator + L1+L2 (34 pairs) | 0.0631 | **0.0516** | 64/68 |
| + AGORA W=0.5 (35 pairs) | 0.0661 | — | 66/70 |
| **+ AGORA W=1.0 (35 pairs)** | **0.0565** | **0.0504** | **70/70** |
| + AGORA W=1.5 (35 pairs) | 0.0597 | — | 66/70 |
| + AGORA W=2.0 (35 pairs) | 0.0575 | — | 70/70 |

</div>
<div class="col">

### Approach
- **10 guilds** (class-level taxonomy, Szafrański et al. 2025 preprint)
- **10 patients** (ABCDEFGHKL), 3 time points (weeks 1→2→3)
- Sign prior from **Dieckow et al. (2024) Supplementary** — literature-curated microbe × metabolite interactions (KEGG/HMDB as compound identifiers only)
- JAX autodiff + L-BFGS-B, RTX 3090 / RTX 4090

</div>
</div>

<div class="box">

**Hypothesis**: Metabolite sign constraints from Dieckow et al. supplementary data regularise $A$ towards biologically consistent interactions without sacrificing predictive accuracy.

</div>

---

## Mathematical Model

**Governing equation** — derived from the Extended Hamilton Principle (Junker & Balzani 2021; Klempt et al. 2024, 2025), 0D homogeneous limit, fully viable cells ($\psi_i=1$), no antibiotic ($\alpha^*=0$):

$$\dot{\phi}_i = \phi_i \!\left( b_i + \sum_j A_{ij}\,\phi_j - \bar{f}(\boldsymbol\phi) \right), \qquad \bar{f}(\boldsymbol\phi) = \sum_k \phi_k f_k(\boldsymbol\phi)$$

where $f_k(\boldsymbol\phi) = b_k + \sum_j A_{kj}\phi_j$ is the fitness of guild $k$.

- $\phi_i(t) \geq 0$, $\sum_i \phi_i = 1$ enforced by the mean-fitness subtraction $\bar{f}$  
- Mathematically equivalent to the replicator equation (Taylor & Jonker 1978; Hofbauer & Sigmund 1998) with linear payoffs $f_i = b_i + \sum_j A_{ij}\phi_j$  
- $b_i$: patient-specific intrinsic growth rate &nbsp;·&nbsp; $A_{ij}$: symmetric interaction matrix ($A_{ij}=A_{ji}$)

**Loss function with metabolite sign penalty**

$$\mathcal{L} = \underbrace{\text{RMSE}(\hat{\boldsymbol\phi}, \boldsymbol\phi)}_{\text{data fit}} + \mathcal{P}_{\text{sign}}(\boldsymbol{A},\boldsymbol{F}) + \lambda\|\boldsymbol{A}\|_F^2$$

$$\mathcal{P}_{\text{sign}} = \sum_{\substack{(i,j)\\F_{ij}\neq 0}} \frac{|F_{ij}|}{2\sigma^2}\left[\max\!\left(0,\,-\operatorname{sgn}(F_{ij})\cdot A_{ij}\right)\right]^2 \quad (\sigma=0.15)$$

---

<!-- _style: "section { font-size: 15px; }" -->

## Mathematical Model — Sign Prior & Optimisation

<div class="cols">
<div class="col">

### Sign prior — data sources (3 layers)
- **L1 (main)** Dieckow et al. (2024) Suppl. — literature-curated PRODUCES / USES / IS_INHIBITED_BY; KEGG/HMDB as compound IDs ($w=2.0$)
- **L2** Additional KEGG-predicted cross-feeding ($w=1.0$)
- **L3 (appendix)** AGORA2 FBA (Heinken et al. 2023, *Nat. Methods*) — pFBA cross-feeding signals on 9 guild SBML models ($w=0.5$)
- Basic model: **22 pairs** (L1+L2) · Expanded: **34 pairs** (L1+L2+L3)

</div>
<div class="col">

### Optimisation
- Warm start from previous fit
- JAX `jax.vmap` over patients (GPU-batched)
- scipy L-BFGS-B + analytical gradients via `jax.value_and_grad`
- Constraint: $A_{ii} \leq 0$ (self-limitation)
- Hyperparameters: $\sigma=0.15$, $\lambda=10^{-4}$ (insensitive; $\sigma\in[0.05,0.30]$ gives identical SA)

</div>
</div>

<div class="box" style="margin-top:20px">

**Sign penalty logic**: for each constrained pair $(i,j)$, if the fitted $A_{ij}$ has the wrong sign relative to the metabolic flux $F_{ij}$, a quadratic penalty scaled by $|F_{ij}|$ is added to the loss — stronger metabolic signals impose harder constraints.

</div>

<div class="box-green" style="margin-top:10px">

**Warm start**: initial $A$ from a fit without prior → prior then regularises towards biologically consistent interactions. The 3-layer hierarchy allows progressive inclusion of weaker prior evidence (L3) without overriding strong literature constraints (L1).

</div>

---

## Result 1 — Interaction Matrix $A$

![center w:950](fig_guild_hamilton_kegg_Amatrix.png)

<div class="box-green">

**Sign agreement = 22/22 (100%)** with Dieckow metabolite prior (22 pairs) &nbsp;·&nbsp; Green ✓ = supported &nbsp;·&nbsp; Red ✗ = discordant &nbsp;·&nbsp; Diagonal $A_{ii}<0$: self-limitation for all guilds

</div>

<div class="cite">

Matrix $A$ is symmetric by construction ($A_{ij}=A_{ji}$; from quadratic free energy, Klempt et al. 2025). Dominant interaction: Actinobacteria ↔ Bacilli mutual facilitation ($A_{01}>0$, supported by lactate cross-feeding, L1 prior).

</div>

---

## Result 2 — Trajectory Prediction (Week 1 → 2 → 3)

![center w:940](fig_slide_trajectory.png)

<div class="metrics" style="margin-top:8px">
  <div class="card"><div class="val">0.874</div><div class="lbl">Pearson <i>r</i> (all patients)</div></div>
  <div class="card"><div class="val">0.0830</div><div class="lbl">Training RMSE</div></div>
  <div class="card"><div class="val">22/22</div><div class="lbl">Metabolite sign agreement</div></div>
  <div class="card"><div class="val">0.987</div><div class="lbl"><i>r</i> — best patient (G)</div></div>
</div>

<div class="cite">Training fit on 8 patients. Prediction: one-step ODE integration (dt = 10⁻⁴, 100 steps/week).</div>

---

## Result 2b — Fitting Results: All 10 Patients

![center w:970](fig_fitting_results.png)

<div class="metrics" style="margin-top:6px">
  <div class="card"><div class="val">0.0631</div><div class="lbl">Training RMSE (all 10 pat.)</div></div>
  <div class="card"><div class="val">0.938</div><div class="lbl">Pearson <i>r</i></div></div>
  <div class="card"><div class="val">64/68</div><div class="lbl">Sign agreement (94%)</div></div>
  <div class="card"><div class="val">34</div><div class="lbl">Prior pairs (L1+L2+L3)</div></div>
</div>

<div class="cite">Hamilton ODE + expanded Dieckow+AGORA prior. Solid bars = observed; hatched (////) = predicted. Per-patient RMSE and Pearson r shown below each panel. Patient A (wk2 RMSE=0.094) is the hardest to fit — Actinobacteria-dominant community not well represented in 9-patient training pool.</div>

---

## Result 3 — LOO-CV: L1+L2 vs AGORA W=1.0

![center w:1000](figs/fig_loo_comparison_agora_vs_l12.png)

<div class="box-green">

**Final LOO-CV results** (10-fold, one patient out): L1+L2 (34 pairs) mean=0.0516 · **AGORA W=1.0 (35 pairs) mean=0.0504** (−2.4%, 7/10 patients improved). Training: RMSE 0.0631→0.0565 (−10%), SA 94%→100%. Patient A hardest in both models (CT2, Actinobacteria outlier). Patient F easiest (CT2, near-zero LOO error).

</div>

<div class="cite">Community types (CT1/CT2) assigned by GMM clustering on week-1 guild fractions. Dieckow et al. (2024) identified two stable community states in 12-patient cohort.</div>

---

## Result 3b — AGORA W=1.0 vs L1+L2: A matrix & b̂ CT analysis

![center w:980](figs/fig_agora_w1p0_analysis.png)

<div class="box-green">

**AGORA W=1.0 b̂ CT comparison**: Actinobacteria shows largest CT1/CT2 difference (+0.35, CT2 higher). Bacilli also CT2>CT1 (+0.16). Betaproteobacteria CT1>CT2 (−0.10). Sign agreement improved 64/68 → 70/70 with AGORA L3 prior. ΔA (right column) shows changes concentrated in Actinobacteria-related interactions.

</div>

---

## Result 4 — Model Comparison (LOO-CV)

![center w:900](fig_guild_model_comparison.png)

<div class="box-green">

**L1+L2 LOO=0.0516 → AGORA W=1.0 LOO=0.0504** (−2.4%, 7/10 patients better) · Train RMSE=0.0565, r=0.951, SA=70/70 (100%) · W=0.5 RMSE=0.0661, W=1.5 RMSE=0.0597, W=2.0 RMSE=0.0575

</div>

<div class="cite">

gLV: $\dot{x}_i = x_i(r_i + \sum_j \alpha_{ij}x_j)$ on $\mathbb{R}_+^n$ — mathematically distinct from replicator on simplex $\Delta^{n-1}$ (Hofbauer & Sigmund 1998, p. 64).

</div>

---

## Key Findings

<div class="cols">
<div class="col">

### What works ✓
- **Full sign consistency**: 22/22 (100%) — Dieckow metabolite prior compatible with Extended Hamilton dynamics
- **Metabolite prior regularises**: LOO-CV 0.0855 → 0.0770 (9.4% improvement)
- **Competitive**: within 3.9% of gLV + metabolite prior (0.0741)
- **Biologically interpretable $A$**: self-limitation $A_{ii}<0$; Actinobacteria ↔ Bacilli facilitation
- **σ robustness**: $\sigma \in [0.05, 0.30]$ all give SA = 22/22, RMSE ≈ 0.081

</div>
<div class="col">

### Remaining gaps
- gLV free still best at **0.0588** — replicator transient approximation limits fit quality on short 3-week window
- Patient A covariate-shift outlier (Actinobacteria 3× mean) dominates generalisation error
- **AGORA W=1.0 (43 pairs): RMSE 0.0565, r=0.951, SA 70/70 (100%)** — best model so far → LOO-CV running

### Equation nomenclature
- **Hamilton ODE** = 0D limit of Extended Hamilton Principle (Junker & Balzani 2021; Klempt et al. 2024, 2025)  
- Mathematically equivalent to replicator eq. (Taylor & Jonker 1978) with linear payoffs

</div>
</div>

---

## Next Steps & Status

| # | Task | Status |
|---|---|---|
| 1 | Replicator + Dieckow prior **LOO-CV** (8-fold, 22 pairs) | ✅ **Done** — 0.0770 |
| 2 | **Expanded flow LOO-CV** (10-fold, ODE, 34 pairs, L1+L2) | ✅ **Done** — LOO mean=**0.0516**, std=0.0211 (A worst: 0.0844, F best: 0.0074) |
| 3 | **AGORA W=1.0 LOO-CV** (10-fold, 35 pairs, L1+L2+L3) | ✅ **Done** — LOO mean=**0.0504**, std=0.0208 (7/10 patients improved vs L1+L2) |
| 4 | **Sign probability** heatmap from 10,000p posterior | ✅ Done |
| 5 | **Paper S1** update with correct posterior statistics | ✅ Done |
| 6 | **σ sensitivity** analysis | ✅ Done — insensitive, σ=0.15 optimal |
| 7 | **Network visualisation** of $A$ matrix | ✅ Done |

<div class="box-green" style="margin-top: 16px">

**Summary**: AGORA W=1.0 best overall: **LOO-RMSE=0.0504** (vs L1+L2 0.0516, −2.4%), training RMSE=0.0565 (vs 0.0631, −10%), SA=70/70 (100%). MacArthur quantitative prior failed (SA=4-8/70). Actinobacteria b̂ CT1/CT2 diff: +0.35. Sign prior weight W=1.0 optimal: phase transition to 100% SA.

</div>

---

## Appendix — 10-Week Extrapolation (AGORA W=1.0)

![center w:1000](figs/fig_agora_w1p0_nweeks_extrapolation.png)

<div class="box">

**MAP extrapolation**: AGORA W=1.0 A matrix + per-patient b̂ → forward integration to week 10. Training range: weeks 1–3 (dots = observed). Shaded region = extrapolation. Week 2 RMSE = **0.061**, week 3 RMSE = **0.040** (MAP, training set). Communities converge toward guild-specific equilibria by week 5–7.

</div>

---

## Appendix — AGORA Weight & Prior Type Sensitivity

![center w:980](figs/fig_agora_weight_sweep.png)

---

## Appendix — AGORA Weight Sensitivity (Table)

| AGORA weight $w$ | Train RMSE | Pearson $r$ | SA (35 pairs) | Notes |
|---|---|---|---|---|
| L1+L2 only (0) | 0.0631 | 0.938 | 64/68 | 34 pairs, baseline |
| **W = 0.5** | 0.0661 | 0.932 | 66/70 | softer L3 prior |
| **W = 1.0** ★ | **0.0565** | **0.951** | **70/70** | optimal |
| **W = 1.5** | 0.0597 | 0.945 | 66/70 | over-constraints |
| **W = 2.0** | 0.0575 | 0.949 | 70/70 | slightly worse |
| MacArthur c=0.01 | 0.0639 | — | 8/70 | Gaussian A~N(c·Φ) failed |
| MacArthur c=0.05 | 0.0592 | — | 4/70 | sign-inconsistent |
| MacArthur c=0.10 | 0.0700 | — | 4/70 | sign-inconsistent |

<div class="box-green">

**W=1.0 optimal**: full sign prior from AGORA2 FBA (weight = Szafrański L2 baseline) achieves best training RMSE and **100% sign agreement** with all 35 constrained pairs. MacArthur quantitative Gaussian prior ($A_{ij} \sim N(c\cdot\Phi_{ij}, \sigma^2)$) fails to enforce sign consistency — ecological signs dominated by data not FBA magnitude.

</div>

---

## Appendix — AGORA2: Sign Prior (L3 Layer)

<div class="cols">
<div class="col">

### 3-layer prior construction

| Layer | Source | Weight |
|---|---|---|
| L1 | Szafrański Suppl. (experimental, KEGG) | 2.0 |
| L1 | Szafrański Suppl. (experimental, other) | 1.5 |
| L2 | Szafrański Suppl. (prediction) | 1.0 |
| **L3** | **AGORA2 pFBA cross-feeding** | **1.0** |

**Cross-feeding signal (j→i)**:  
guild $j$ secretes $\alpha$ ∧ guild $i$ takes up $\alpha$ → `net_flow[i,j] += 1.0`

**Prior pairs: 11 → 43** (L1+L2 → +L3)

### Validation

**92% sign agreement** (66/72 pairs) between AGORA2 FBA and fitted gLV $A$

</div>
<div class="col">

### Guild representatives (AGORA2 v2.01)

| Guild | Strain |
|---|---|
| Actinobacteria | *A. naeslundii* Howell 279 |
| Bacilli | *S. gordonii* CH1 |
| Negativicutes | *V. parvula* Te3 DSM 2008 |
| Bacteroidia | *P. melaninogenica* ATCC 25845 |
| Fusobacteriia | *F. nucleatum* ATCC 25586 |
| Clostridia | *P. micra* ATCC 33270 |
| Coriobacteriia | *Atopobium parvulum* DSM 20469 |
| Flavobacteriia | *Capnocytophaga sputigena* ATCC 33612 |
| β/γ-Prot. | *Eikenella* / *Haemophilus* |

**Heinken et al. 2023** *Nat. Biotechnol.* 41:1320  
DOI: 10.1038/s41587-022-01628-0

</div>
</div>

---

## Appendix — AGORA2: FBA Pipeline Detail

<div class="cols">
<div class="col">

### Software stack

| Tool | Version | Role |
|---|---|---|
| **cobrapy** | 0.31 | SBML I/O, FBA solver interface |
| **libsbml** | — | AGORA2 SBML parsing |
| **scipy** | — | LP backend (linprog) |
| **AGORA2** | v2.01 | 7,302 GEMs, vmh.life |

### pFBA per guild

```python
model = read_sbml_model("guild.xml")   # cobrapy

# Apply oral-fluid medium
# Note: AGORA2 uses EX_glc_D(e)
#       not BiGG modern EX_glc__D_e
model.medium = ORAL_MEDIUM

# Parsimonious FBA
# maximise growth, then minimise Σ|flux|
sol = pfba(model)

secretion = {met: f for f > 0}   # → env
uptake    = {met: |f| for f < 0} # ← env
```

</div>
<div class="col">

### Oral-fluid medium composition

| Category | Metabolites |
|---|---|
| Carbon sources | glucose, fructose, sucrose, lactate |
| Amino acids | 20 standard + Cys, Orn |
| Vitamins B | B₁, B₂, B₃, B₅, B₆ (×3 forms), B₁₂ |
| Cofactors | heme, siroheme, adenosylcobalamin, menaquinone, ubiquinone |
| Cell wall | **meso-DAP** (peptidoglycan precursor) |
| Lipids | stearate C18, myristate C14, laurate C12 |
| Trace metals | Fe²⁺/³⁺, Mg²⁺, Ca²⁺, Zn²⁺, **Cu²⁺**, Mn²⁺, Co²⁺ |
| Other | glutathione (ox/red), putrescine, spermidine, ornithine, cytidine |

**Calibration**: Cu²⁺, meso-DAP, B₆ pyridoxine, cell-wall lipids were added iteratively to achieve $\mu > 0$ for all 10 guilds (verified per model).

All 10 guilds: $\mu \in [0.11, 1.66]\ \text{h}^{-1}$ ✓

</div>
</div>

---

## References

<div style="font-size:16px; line-height:1.6">

**Dataset** &nbsp; Dieckow S, Szafrański SP et al. (2024) *npj Biofilms Microbiomes* 10:85. doi:10.1038/s41522-024-00624-3

**Metabolite prior (L1, main)** &nbsp; Dieckow S, Szafrański SP et al. (2024) Suppl. File 1 — literature-curated microbe × metabolite interactions (KEGG/HMDB compound IDs)

**AGORA2 (L3, appendix)** &nbsp; Heinken A et al. (2023) *Nat. Methods* 20:1022–1035. doi:10.1038/s41592-023-01919-1

**Extended Hamilton Principle** &nbsp; Junker P & Balzani D (2021) *Comput. Methods Appl. Mech. Eng.* 380:113773; Junker P, Bode M & Hackl K (2025) *J. Mech. Phys. Solids*

**Hamilton ODE for biofilm** &nbsp; Klempt F et al. (2024) *Biomech. Model. Mechanobiol.* — Continuum growth; Klempt F et al. (2025) — Continuum biofilm (preprint)

**Replicator equation** &nbsp; Taylor PD & Jonker LB (1978) *Math. Biosci.* 40:145–156; Hofbauer J & Sigmund K (1998) *Evolutionary Games and Population Dynamics*. Cambridge Univ. Press

**Optimisation** &nbsp; JAX (Bradbury et al. 2018, github.com/jax-ml/jax); scipy L-BFGS-B (Byrd et al. 1995)

**Compound identifiers** &nbsp; KEGG: Kanehisa M & Goto S (2000) *Nucleic Acids Res.* 28:27–30 · HMDB: Wishart DS et al. (2022) *Nucleic Acids Res.* 50:D622–D631

</div>
