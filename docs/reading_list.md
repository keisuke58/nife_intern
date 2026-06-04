# Reading list

Curated key references for the NIFE oral-biofilm modelling project, plus an auto-appended literature watch.
Add new finds with `/litwatch` (on-demand arXiv scan — gentle, one-shot, never a loop). Edit the curated section
by hand.

> Constraints (shared research-lab server): `/litwatch` does read-only checks + one file append only. It must not
> submit cluster jobs, launch GPU compute, run heavy compute on the login node, or hammer the network. Keep it light.

## Curated references (grouped by theme)

### Genome-scale metabolic / cross-feeding communities (the L3 sign prior, COMETS/dFBA pillar)
- **Heinken et al. 2023, *Nat Biotechnol*** — AGORA2: genome-scale reconstructions for >7000 human-microbiome
  strains. Source of the metabolic models behind the L3 cross-feeding sign prior and the COMETS/dFBA validation.
- **Diener et al. 2020, *mSystems*** — MICOM: metabolic interaction modelling of microbial communities (community
  FBA with growth-rate trade-off). Cross-validates the ecological interactions against community-level fluxes.
- **MacArthur 1970, *Theor Popul Biol*** — consumer-resource theory; the mechanistic underpinning that connects
  resource competition to effective Lotka-Volterra interaction coefficients.
- **Marsland et al. 2019, *PLoS Comput Biol*** — the Community Simulator: a consumer-resource framework linking
  metabolism to emergent community structure; conceptual bridge from FBA cross-feeding to gLV-style interactions.

### Ecological inference — gLV / replicator / Hamilton (the paper)
- **Taylor & Jonker 1978, *Math Biosci*** — evolutionarily stable strategies and replicator dynamics; the
  replicator form (composition sums to 1) used by both the gLV and Hamilton models.
- **Junker & Balzani 2021; Klempt et al. 2024–2025** — Hamilton-principle formulation of dissipative/ecological
  dynamics; basis of the symmetric-A JAX Hamilton model used alongside the asymmetric gLV.

### Spatial reaction-diffusion / cross-diffusion / PINN (the spatial PDE pillar)
- **Painter & Hillen 2002, *Can Appl Math Q*** — volume-filling chemotaxis / cross-diffusion; the mechanism for
  density-dependent cross-diffusion in the spatial biofilm PDE.
- **Freingruber et al. 2025** — cross-diffusion in generalized Lotka-Volterra systems; directly motivates the
  cross-diffusion term in the spatial gLV/PDE extension.
- **Raissi et al. 2019, *J Comput Phys*** — physics-informed neural networks (PINNs); the method for the inverse
  reaction-diffusion fit (recovering D and reaction params from the Heine HOBIC / FISH spatial data).

### Datasets / domain (oral microbiome, dysbiosis, peri-implantitis)
- **Szafrański 2025, *mSystems*** — 127 cross-sectional oral 16S samples, 5 genera; attractor / community-type
  comparison and source of the L1/L2 experimental + predicted interaction signs.
- **Dieckow 2024 (PRJEB71108)** — 10 patients × 3 weeks longitudinal 16S; the primary fit/LOO-CV dataset.
- **Heine 2025** — in-vitro ODE system; the four community attractors (CS/CH/DS/DH) and the HOBIC spatial data.
- **Botelho 2021 (PRJNA725874)** — 15 patients × 7 timepoints longitudinal 16S; longer time-series validation.
- **Joshi 2025** — oral microbiome dysbiosis / peri-implantitis reference for the clinical framing.

## New (auto-appended by /litwatch)

### 2026-06-04 (litwatch)

- **Sakanaka A, Kuboniwa M, Shimma S, et al. 2022, *mSystems*** — F. nucleatum channels arginine through ornithine to putrescine, forming a trophic hub between commensal S. gordonii and pathogen P. gingivalis, accelerating dysbiotic biofilm formation — directly supports the Fn guild's bridging role in the HOBIC community model.
- **Zapién-Campos R, Bansept F, Traulsen A 2024, *PLOS Biology*** — Stochastic LV inference matching both means and higher moments of abundance recovers true interaction parameters more accurately than deterministic fitting in the low-replicate regime — directly relevant to the 10-patient Dieckow cohort.
- **Castro M, Vida R, Galeano J, Cuesta JA 2025, *Journal of The Royal Society Interface*** — Bayesian assessment of gLV on short ecological time series reveals systematic overfitting and non-interpretable interaction matrices under limited data, arguing for information-theoretic model complexity reduction — a direct methodological caveat for small-cohort oral microbiome inference.
- **Zhang Y, Li Y, Yang Y, et al. 2022, *Frontiers in Microbiology*** — Network co-occurrence analysis of 64 subgingival plaque samples showed dysbiosis is characterised by loss of hub-species connectivity, providing a network-ecology framework directly applicable to guild-level gLV modelling of peri-implantitis.
- **Joshi AA, Szafrański SP, Steglich M, et al. 2025, *Journal of Dental Research*** — Full-length 16S and metatranscriptomics of 49 peri-implantitis implants (SIIRI/SFB TRR-298 cohort) revealed severity-specific signatures including positive correlation of Pseudoramibacter and negative correlation of Capnocytophaga with probing depth.
- **Jiang SS, Chen YX, Fang JY, et al. 2025, *Nature Reviews Microbiology*** — Comprehensive review of F. nucleatum's role as bridge organism in oral biofilm, virulence factors, immune evasion, and cross-site dissemination — the most current single reference for motivating the Fn guild's central position in the HOBIC/dysbiotic model.
- **Huang Y, Tang T, Dai X, Sun F 2025, *PLOS Computational Biology*** — The iterative LV model adapts gLV inference to relative (compositional) 16S data via iterative linearisation, outperforming compositional-LV and standard gLV for recovering interaction coefficients from sparse microbiome time series.
- **Paredes-Vázquez A, Balsa-Canto E, Banga JR 2025, *PLOS Computational Biology*** — Four-stage workflow — structural identifiability, global optimisation, stability check, predictive power — diagnoses and resolves pitfalls in nonlinear ODE calibration from sparse longitudinal microbiome data.
- **Pasqualini J, Maritan A, Rinaldo A, Facchin S, Savarino EV, Altieri A, Suweis S 2024, *eLife*** — Statistical-physics tools applied to disordered gLV show that healthy vs. dysbiotic microbiomes differ in the statistical structure of inter-species interaction networks, providing a theoretical basis for the healthy/dysbiotic attractor dichotomy (CS/CH vs. DS/DH).
- **Espinoza-Arrue J, Arce M, Endo N, et al. 2025, *Journal of Clinical Periodontology*** — 16S profiling across peri-implant health/mucositis/peri-implantitis in a Chilean cohort showed each clinical state has a distinct community structure and co-occurrence network, with peri-implantitis exhibiting enriched pathogenesis pathways — supports the attractor-state framing.

**Gaps identified (themes with no new papers retrieved this session):** spatial PDE / reaction-diffusion; COMETS / dynamic FBA / dFBA; TMCMC / Sequential Monte Carlo Bayesian inference; network analysis / graph-theoretic community ecology (only indirect coverage via Zhang 2022); metabolic sign priors / genome-scale FBA cross-feeding (Heinken 2023 AGORA2 is already in the curated list); FISH / confocal imaging / CLSM analysis; compositional data analysis / Aitchison geometry beyond gLV; stability / resilience theory for microbial communities (indirect coverage via Pasqualini 2024).

<!-- /litwatch appends a `### YYYY-MM-DD` subheading with one bullet per new arXiv paper below this line. -->
