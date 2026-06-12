# Böl 2014 — prior art & differentiator paragraph (DRAFT for ch5 "Validation & limits")

> **Thesis-freeze respected** — this is a draft for insertion *on approval*; nothing in
> `30_Masterarbeit` is edited. Insert near the end of ch5's FEM section (Validation & limits, §6 of
> INTEGRATION_PLAN), right where we concede the γ-magnitude is not yet mechanically validated.

## The paper
Bolea Albero, A., Ehret, A. E., & Böl, M. (2014). *A new approach to the simulation of microbial
biofilms by a theory of fluid-like pressure-restricted finite growth.* **Computer Methods in Applied
Mechanics and Engineering, 272, 271–289.** https://doi.org/10.1016/j.cma.2014.01.001

**What they do:** extend the classical multiplicative split **F = Fₑ·F_g** and establish a formal
connection between finite growth and **finite viscoelasticity**. New material is added by *isotropic
deposition*; the resulting residual stresses are **continuously relieved** by a viscous (fluid-like)
mechanism whose rate depends on the local **hydrostatic pressure** — so the biofilm flows to fill the
available volume ("pressure-restricted" growth) and residual stress does not accumulate without bound.
Single-phase EPS continuum; growth is phenomenological (deposition), not species-resolved.

## Draft paragraph (differentiator — examiner-proof framing)

> The mechanical framework used here is closest in spirit to the finite-growth biofilm models of Böl
> and co-workers, who likewise build on the multiplicative decomposition F = Fₑ·F_g and even derive
> the resulting residual-stress field directly from confocal biofilm geometries [Böl et al., 2009;
> Bolea Albero, Ehret & Böl, 2014]. Two differences delimit the contribution of this chapter. First,
> in those models growth is a **phenomenological isotropic deposition** of new EPS, and the attendant
> residual stress is **continuously relaxed** by a pressure-restricted, fluid-like viscous flow; here
> growth is instead driven by a **species-resolved field, φ_Pg(z)**, obtained from the Hamilton-
> principle ecological model and calibrated against CLSM depth profiles, and is **anisotropic and
> substratum-normal** (F_g = diag(1,1,λ_z(φ_Pg))) rather than isotropic. The mechanical state we
> report is therefore the *composition-resolved* residual stress that a given dysbiotic community
> structure imprints on the film — a quantity the deposition models, which are blind to who grows
> where, cannot produce. Second, we deliberately model the **fast, elastic limit** (a compressible
> neo-Hookean Fₑ): on the seconds-to-minutes timescale over which the depth structure is mechanically
> interrogated, biofilms respond elastically [Stoodley et al., 2002], so the residual stress computed
> here is the short-time stress *before* viscous relaxation. The fluid-like relaxation of Bolea Albero
> et al. is exactly the long-time complement of our model, and incorporating it (a Prony/Maxwell
> relaxation of Fₑ, with the shear relaxation spectrum taken from biofilm rheometry — G ≈ 0.2–24 Pa,
> η ≈ 10–3000 Pa·s [Shaw et al., 2004]) is the natural viscoelastic extension set out in the Outlook.
> Consequently we present the residual-stress magnitude as **calibrated in shape (through φ_Pg(z)) but
> parametric in scale (through γ)**: it is not yet validated against a direct mechanical measurement,
> and the validation pathway — AFM nanoindentation of depth-sectioned oral biofilm [e.g. PMC8355335],
> region-specific micro-rheology, or shear-flow detachment-onset versus P. gingivalis load — is left
> as the decisive next experiment.

## Citations to add to the .bib
- Bolea Albero, Ehret, Böl (2014) CMAME 272:271–289.
- Böl, Möhle, Haesner (2009) *3D FE model of biofilm detachment …*, Biotechnol. Bioeng. 104(1).
- Stoodley, Cargo, Rupp, Wilson, Klapper (2002) J. Ind. Microbiol. Biotechnol. 29:361–367.
- Shaw, Winston, Rupp, Klapper, Stoodley (2004) *Viscoelastic properties … rheometer creep*, Biofouling.
- (oral elastic) Dependency of hydration & growth conditions on oral-biofilm mechanics, PMC8355335.
- (bonus, visco-growth FEM) *Finite strain visco-elastic growth driven by nutrient diffusion … biofilm*,
  Comput. Mech. 2019, doi:10.1007/s00466-019-01708-0 — supports the Outlook viscoelastic-growth route.
