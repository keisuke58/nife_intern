# External resources — datasets · papers · git repos (2026-06-13)

Curated to fill the documented gaps in `MISSING_ITEMS_ROADMAP.md` / `EXTENSIONS_RECIPES.md`:
③ validation (mechanical measurement), ④ constitutive justification (G, viscoelastic),
B3 viscoelasticity, and reusable FEM-growth code for the F=Fe·Fg UMAT.

Legend: **[must-cite]** examiner-critical · **[reuse]** code we can build on · **[data]** numbers/values.

---

## 0. PRIOR ART — read first (examiner-proofing)
This thesis = "composition-driven growth → residual stress in a biofilm, via FE". **Markus Böl &
Alexander Ehret (TU Braunschweig / Empa) already published almost exactly this conceptual chain.**
Must cite and must state our differentiator (ours: *species-resolved* growth driver φ_Pg(z) from
Hamilton-principle ecology + CLSM calibration; theirs: phenomenological isotropic deposition).

- **[must-cite]** Böl et al. 2009, *3D finite element model of biofilm detachment using real biofilm
  structures from CLSM data*, Biotechnol. Bioeng. — **CLSM → FEM, the exact pipeline of our ch5.**
  https://www.ncbi.nlm.nih.gov/pubmed/19191328
- **[must-cite]** *A new approach to the simulation of microbial biofilms by a theory of fluid-like
  pressure-restricted finite growth*, J. Mech. Phys. Solids / Mech. Mater. 2014 — **finite growth with
  residual stresses from isotropic deposition, continuously relaxed.** Closest prior art to our
  residual-stress story; differentiate explicitly.
  https://www.sciencedirect.com/science/article/abs/pii/S004578251400005X
- **[must-cite]** Ehret & Böl, *Material modeling of biofilm mechanical properties*, Math. Biosci. 2014
  — WLC-network constitutive model for EPS (our neo-Hookean is the cruder fast-elastic limit; cite as
  the more physical option for Outlook ④). https://www.sciencedirect.com/science/article/abs/pii/S0025556414000339
- Li et al. 2020, *Predicting biofilm deformation with a viscoelastic phase-field model*, Biotechnol.
  Bioeng. — viscoelastic + experiment; supports B3.
  https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/abs/10.1002/bit.27491

---

## 1. ④ Constitutive justification — modulus & viscoelastic VALUES
Key nuance to put in the thesis: **modulus is measurement-timescale dependent** — bulk creep rheology
gives G ~ O(1–10 Pa) (long-time, fluid-like); AFM nanoindentation gives E ~ O(1 MPa) (short-time,
local elastic). Our neo-Hookean Fe is the fast/instantaneous limit → MPa-range E defensible for the
elastic part; report σ/μ to stay modulus-agnostic.

- **[data][must-cite]** Stoodley et al. 2002, *Biofilm material properties as related to shear-induced
  deformation and detachment*, J. Ind. Microbiol. Biotechnol. — viscoelastic fluid; elastic over
  seconds, viscous over longer. https://pubmed.ncbi.nlm.nih.gov/12483479/
- **[data]** Shaw / Stoodley *Viscoelastic properties of a mixed culture biofilm from rheometer creep
  analysis*, Biofouling 2004 — **4-element Burger model: G = 0.2–24 Pa, η = 10–3000 Pa·s.** Direct
  numbers for a Prony/Maxwell B3 fit. https://pubmed.ncbi.nlm.nih.gov/14650082/
- **[data]** *Dependency of hydration and growth conditions on the mechanical properties of oral
  biofilms*, 2021 (PMC8355335) — **ORAL biofilm, AFM: E ≈ 1.39–2.83 MPa**, hydration-dependent.
  Best oral-specific elastic number. https://pmc.ncbi.nlm.nih.gov/articles/PMC8355335/
- **[data]** *A Multi-scale Biophysical Approach to Structure–Property Relationships in Oral Biofilms*,
  Sci. Rep. 2018 — multi-scale oral-biofilm mechanics. https://www.nature.com/articles/s41598-018-23798-1
- **[data]** *Regulating, Measuring, and Modeling the Viscoelasticity of Bacterial Biofilms*, J.
  Bacteriol. 2019 — review of G(t), creep/relaxation, Maxwell/Burgers. https://journals.asm.org/doi/10.1128/jb.00101-19
- *Perspective: viscoelastic properties of biofilm infections…*, 2023 (PMC9978752) — recent review,
  good for framing. https://pmc.ncbi.nlm.nih.gov/articles/PMC9978752/

## 2. ③ Validation pathway (what would validate the γ-magnitude)
No public "biofilm residual-stress" dataset exists (state this honestly). Validation routes, each with
method precedent above:
- AFM nanoindentation maps (PMC8355335 method) → local E(z) to anchor μ.
- Micro-rheology region-specific (PMC5460257) → depth-varying G for a graded model.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5460257/
- Shear-flow detachment-onset vs Pg load → tests the B2 delamination claim.

## 3. Reusable git repos (for the F=Fe·Fg UMAT / B-series)
- **[reuse]** `mholla/growth` — Abaqus UMAT subroutines for growth (Holland, *Hitchhiker's Guide to
  Abaqus*). Runnable UMAT+inp pairs for volumetric/anisotropic growth = direct template for our
  USERMAT/UMAT finite-growth port. https://github.com/mholla/growth
- **[reuse]** `jpsferreira/UMAT-ABAQUS` — general framework to develop UMAT material models (clean
  scaffolding, AD tangent patterns). https://github.com/jpsferreira/UMAT-ABAQUS
- **[reuse]** `victorlefevre/UMAT_Lefevre_Sozio_Lopez-Pamies` — finite viscoelasticity UMAT family →
  template for B3. https://github.com/victorlefevre/UMAT_Lefevre_Sozio_Lopez-Pamies
- **[reuse]** `tajtac/Anisotropic_NN_UMAT` — NN/surrogate material model with Abaqus UMAT → mirrors our
  offline-tabulated-surrogate path (umat_server_fs_tab.py). https://github.com/tajtac/Anisotropic_NN_UMAT
- **[reuse]** `febiosoftware/FEBio` — open biomech FE solver with native growth/constitutive models;
  cross-check / alternative to Abaqus. https://github.com/febiosoftware/FEBio
- **[reuse, FEniCS]** `annekerachni/braingrowthFEniCS` — multiplicative growth split F=Fe·Fg in FEniCS,
  CLSM/MRI-informed → analogous depth-graded growth pattern; good open reference for the method chapter.
  https://github.com/annekerachni/braingrowthFEniCS

## 4. Theory refs for the morphoelastic method chapter
- *The multiplicative deformation split for shells … growth, swelling, thermo-/visco-elasticity*,
  arXiv:1810.10384. https://arxiv.org/pdf/1810.10384
- *Chemomechanical regulation of growing tissues from a thermodynamically-consistent framework*
  (tumor spheroid), PMC11643224 — growth-rate ↔ stress coupling, supports our γ(φ_Pg) closure.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11643224/

## 5. Follow-up artifacts produced from this search (2026-06-13)
- `notes/mholla_growth_diff.md` — diff of mholla/growth vs our kernel; portable parts (geometric
  tangent terms we lack; `umat_fiber_morph.f` n=e_z = our depth growth; socket-free USERMAT plan).
- `coupling_prototype/prony_from_burger.py` (+ `prony_biofilm.json`) — exact 2-term Burgers→Prony
  (g1=0.338@6.6s, g2=0.662@903s for the default set) and the `*VISCOELASTIC, TIME=PRONY` block.
- `coupling_prototype/abaqus/biofilm_viscoelastic.inp` — B3 one-element shear-relaxation deck.
- `notes/bol2014_validation_paragraph.md` — Böl 2014 summary + ch5 "Validation & limits" draft
  paragraph (differentiator: species-resolved φ_Pg growth vs their phenomenological deposition).
- Bonus ref: *Finite strain visco-elastic growth driven by nutrient diffusion … biofilm*, Comput.
  Mech. 2019, doi:10.1007/s00466-019-01708-0 — viscoelastic-growth FEM, Outlook support.

---
### Bottom line
- **Must-cite & differentiate**: Böl/Ehret biofilm finite-growth + CLSM-FEM line (§0) — closest prior
  art; our edge = species-resolved Hamilton-ecology growth driver.
- **④ filled**: Burger G=0.2–24 Pa / η=10–3000 Pa·s (rheology) and oral E=1.39–2.83 MPa (AFM); use the
  timescale split to justify neo-Hookean Fe.
- **Reuse now**: `mholla/growth` as the UMAT growth template; `braingrowthFEniCS` as the open
  multiplicative-growth reference.
- **③ honest**: no residual-stress dataset exists; list AFM/micro-rheo/detachment as validation routes.
