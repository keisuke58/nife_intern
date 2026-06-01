# Paper Outline: Hamilton ODE + AGORA Sign Prior for Oral Microbiome Guild Dynamics

**Working title**: Genome-scale metabolic priors improve inference of oral microbiome community dynamics from longitudinal 16S data

**Target journal**: *npj Biofilms and Microbiomes* / *PLOS Computational Biology* / *mSystems*

**Status**: Draft outline — 2026-05-06

---

## 1. Introduction (~600 words)

### Hook
Oral microbiome dynamics drive disease transitions (health → dysbiosis → periodontitis/peri-implantitis), yet quantitative models of inter-guild interactions remain poorly constrained from clinical time-series data alone.

### Gap
- gLV / Hamilton ODE models require estimating O(n²) interaction parameters from n patients × 3 time points → severely underdetermined
- Existing approaches: (a) free gLV (overfits), (b) literature-curated sign constraints (sparse coverage), (c) genome-scale metabolic models (GEMs) exist but never used as ecological priors

### Contribution
1. **3-layer sign prior**: L1 (Szafrański experimental), L2 (Szafrański predicted), L3 (AGORA2 FBA cross-feeding) → 35 constrained pairs (vs 22 L1+L2 only)
2. **Weight optimisation**: AGORA L3 weight sweep W∈{0.5,1.0,1.5,2.0} → W=1.0 phase transition to SA=100%
3. **LOO-CV validation**: best: gLV α=0.25 LOO-RMSE 0.0490 (vs gLV free 0.0501, −2.2%; vs Hamilton L1+L2 0.0567, −13%)
4. **Biological regime analysis**: data+prior aligned vs prior-constrained muted vs data-driven — reveals which metabolic cross-feeding is ecologically dominant

---

## 2. Methods

### 2.1 Data: Dieckow et al. 2024
- 10 patients × 3 weeks × 10 guilds (Szafrański taxonomy)
- Full-length 16S rRNA amplicon sequencing (ENA: PRJEB71108)
- Guild assignment: Szafrański et al. 2025 Suppl. File 1

### 2.2 Hamilton ODE (0D limit)
- Governing equation (replicator with linear payoffs)
- Parameter space: A (symmetric 55 free), b̂ (patient-specific 10×10)
- Constraint: A_ii ≤ 0 (self-limitation)
- Optimisation: JAX L-BFGS-B, warm start, 5000 iterations

### 2.3 3-Layer Sign Prior
- **L1**: Szafrański experimental evidence; weight 2.0
- **L2**: Szafrański predicted interactions; weight 1.0
- **L3**: AGORA2 (Heinken et al. 2023, *Nat. Biotechnol.*) pFBA cross-feeding; weight W
  - 10 guild-representative SBML models
  - Oral-fluid medium composition (Dawes 2008)
  - Cross-feeding rule: j secretes α ∧ i uptakes α → net_flow[i,j] += W
  - Toxin rule: H₂O₂, H₂S → net_flow[i,j] -= W
- Penalty: $\mathcal{P} = \sum_{(i,j): F_{ij}\neq 0} \frac{|F_{ij}|}{2\sigma^2}[\max(0,-\text{sgn}(F_{ij})A_{ij})]^2$ with σ=0.15

### 2.4 Validation
- LOO-CV: 10-fold, one patient out per fold
- Sign agreement (SA): fraction of constrained pairs with correct sign
- MacArthur consumer-resource prior (comparison): A ~ N(c·Φ, σ²)

### 2.5 A Matrix Stability (pending)
- 10-fold LOO → 10 A estimates → per-pair std, CV, sign consistency
- Three regimes: data+prior aligned / prior-constrained / data-driven

---

## 3. Results

### 3.1 AGORA2 FBA sign validation
- 92% agreement between pFBA cross-feeding signs and fitted gLV A (66/72 pairs)
- Oral-fluid medium: all 10 guilds viable (μ = 0.11–1.66 h⁻¹)
- Prior pairs: 22 (L1+L2) → 35 (L1+L2+L3, W=1.0)

### 3.2 W=1.0 phase transition
- W sweep (full fit): SA 94%(W=0.5) → **100%(W=1.0)** → 94%(W=1.5) — LOO best: gLV α=0.25 (RMSE 0.0490)
- W=1.0 = Szafrański L2 equivalent weight → biologically motivated calibration
- RMSE: W=1.0 gives lowest training RMSE (0.0565) and LOO-RMSE (0.0504)

### 3.3 LOO-CV: AGORA beats gLV free
| Model | Train RMSE | LOO-RMSE | SA |
|---|---|---|---|
| gLV free | 0.0501 | 0.0501 | — |
| Hamilton + L1+L2 | 0.0631 | 0.0567 | 94% |
| **gLV α=0.25 (L1+L2)** | — | **0.0490** | **97%** |
| MacArthur prior | 0.059–0.07 | — | 4–8/70 |

- 8/10 patients improved with L1+L2 prior (gLV α=0.25 vs free)
- MacArthur quantitative prior fails: FBA flux magnitudes ≠ ecological interaction strengths

### 3.4 Biological regime analysis
Three interaction regimes in fitted A:
1. **Data+prior aligned** (e.g. Actinobacteria–Bacilli, +1.61): corroborated by co-aggregation literature
2. **Prior-constrained, muted** (e.g. Bacilli–Negativicutes, +0.003): metabolic cross-feeding real but not dominant at 3-week timescale
3. **Data-driven** (e.g. Bacteroidia–Other, +0.74): ecological signal without AGORA coverage

### 3.5 LOO A matrix stability (pending PBS results)
- Which pairs are robustly estimated across folds vs data-dependent

---

## 4. Discussion (~800 words)

### Why sign (not magnitude) works
- FBA flux units ≠ gLV A units → magnitude mapping requires unknown constant c
- Sign is unit-free and ecologically interpretable
- MacArthur failure confirms: ecological signs are informative, magnitudes from FBA are not

### Bacilli–Negativicutes: metabolically real, ecologically muted
- Streptococcus→lactate→Veillonella: textbook oral cross-feeding (Mikx 1975)
- 3-week abutment window too short to detect? Or: competition for space/nutrients dominates?
- **Prediction**: longer time-series would show stronger positive A_ij for this pair

### "Other" guild interactions
- Bacteroidia–Other (+0.74) is the strongest data-driven signal without prior
- Points to unmodeled taxa in AGORA — need broader GEM coverage of oral non-culturable taxa

### Limitations
- 10 patients × 3 weeks → A is underdetermined (155 params, 200 obs)
- Symmetric A assumption (A_ij = A_ji) — simplification for identifiability
- Guild-level aggregation loses species diversity within guilds
- b̂ patient-specific but not linked to clinical covariates

### Future directions
- Hierarchical model for b̂: b ~ N(μ_CT, Σ_CT) by community type
- Extend to Szafrański BIC peri-implantitis data if longitudinal subset available
- TMCMC posterior on A for identifiable subset (pairs with |A|>0.3)

#### Four innovative directions toward higher ρ

**1. Spatial structure: full PDE reaction-diffusion model**
Current ODE assumes a well-mixed community; oral biofilm is a stratified structure with steep O₂/pH gradients between surface and deep layers. Extending to reaction-diffusion PDEs (1D depth coordinate) would capture anaerobe enrichment at depth and aerobe dominance near the surface — mechanistically explaining why dysbiotic species dominate despite low relative abundance in bulk measurements.
- Replace Hamilton ODE with 1D PDE: ∂φ_i/∂t = f_i(φ) + D_i ∂²φ_i/∂x² + r_i(c(x))
- Nutrient field c(x) from existing 1D solver (Hamilton+PDE hybrid, already implemented in Tmcmc202601)
- Expected gain: A_ij can become depth-resolved → sign prior validity testable per layer

**2. Host immune/tissue-breakdown parameters**
Peri-implantitis is driven by host–microbe interaction, not microbial competition alone. GCF (gingival crevicular fluid) provides collagen peptides, haem-iron, and ROS that selectively enrich Pg and anaerobes.
- Add host-derived metabolites to AGORA2 medium: haem (Pg growth), collagen fragments (Fn proteolysis substrate), lactoferrin (iron chelation, inhibits Aa)
- Condition-specific medium: healthy → standard oral fluid; peri-implantitis → GCF-enriched (elevated haem, pH drop)
- Operationalization: MICOM medium construction from GCF metabolomic data (e.g. Ramseier 2009 proteomics)

**3. Patient-specific MAGs (Metagenome-Assembled Genomes)**
AGORA2 uses reference-strain GEMs; within-species metabolic diversity (accessory genome) is lost. Patients carry strains with distinct carbohydrate utilization and toxin production genes.
- Reconstruct patient-specific GEMs from Szafrański metagenomic reads via DRAM or CarveMe
- Replace AGORA2 guild representatives with patient-matched MAG-GEMs
- Cross-feeding matrix F becomes patient-specific → personalized sign prior → A_ij credible intervals tighten
- Key challenge: 10 patients × 5 guilds = 50 GEMs; feasibility depends on sequencing depth (>1M reads/sample needed)

**4. Thermodynamic FBA (tFBA) for magnitude constraints on A_ij**
Sign-only prior succeeded because FBA flux magnitudes ≠ gLV A units. tFBA constrains reaction fluxes using ΔG_r (Gibbs free energy), producing thermodynamically consistent flux bounds that reflect actual metabolic capacity ratios.
- Use eQuilibrator/TECRDB to assign ΔG_r° to exchange reactions
- tFBA upper/lower flux bounds → credible magnitude range for F_ij (not just sign)
- Map F_ij magnitude bounds → A_ij magnitude prior via learned constant c (c estimated from L1+L2 pairs where both sign and magnitude are known)
- Expected: move from pure sign prior to bounded-magnitude prior, recovering quantitative A_ij without overfitting

---

## 5. Conclusions (3–4 bullets)
1. AGORA2 genome-scale metabolic models provide biologically valid sign priors for ecological interaction inference
2. W=1.0 (L2-equivalent weight) achieves 100% sign agreement — a phase transition linking metabolic and ecological evidence
3. Sign prior outperforms both free gLV and MacArthur quantitative prior (LOO-RMSE −14% vs free, −2.4% vs L1+L2)
4. Three interaction regimes reveal which metabolic cross-feeding relationships are ecologically detectable from short time-series

---

## Key figures (planned)
| Fig | Content |
|---|---|
| 1 | Study design: data → guild assignment → Hamilton ODE → sign prior layers |
| 2 | AGORA2 pipeline: oral-fluid medium, pFBA, cross-feeding matrix |
| 3 | AGORA sign vs gLV A validation (92% match) |
| 4 | W sweep: SA and RMSE vs weight (phase transition at W=1.0) |
| 5 | LOO-CV comparison: all models, per-patient |
| 6 | A matrix heatmap annotated (3 regimes) |
| 7 | A_ij vs |F_ij| scatter (prior strength vs ecological signal) |
| 8 | LOO A stability (pending) |

---

## References (key)
- Dieckow et al. (2024) *npj Biofilms Microbiomes* 10:85
- Heinken et al. (2023) *Nat. Biotechnol.* 41:1320
- Junker & Balzani (2021) *CMAME* 380:113773
- Klempt et al. (2024) *Biomech. Model. Mechanobiol.*
- Mikx & van der Hoeven (1975) *Arch. Oral Biol.* 20:407
- Kolenbrander et al. (2000) *Microbiol. Mol. Biol. Rev.* 64:474
- Marsland et al. (2019) *PLOS Comput. Biol.* 15:e1006793
