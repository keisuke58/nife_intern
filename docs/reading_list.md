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

<!-- /litwatch appends a `### YYYY-MM-DD` subheading with one bullet per new arXiv paper below this line. -->
