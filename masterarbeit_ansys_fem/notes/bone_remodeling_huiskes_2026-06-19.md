# Bone remodeling — Huiskes/Frost mechanostat addition (2026-06-19)

Adds the "standard language" of peri-implant FEM (Huiskes 1987 / 2000 site-specific apparent-
density remodeling, with the Frost 1987 mechanostat lazy zone) on top of the project's existing
mechanistic RANKL/OPG disease model. The two are *mechanistically equivalent at steady state* under
a TGF-β → osteoblast / RANKL → osteoclast mapping, so the thesis can speak both languages without
re-doing the disease model:

  Huiskes language ─────────  used by ~80 % of dental-implant FEM literature → examiner-friendly
  RANKL/OPG language ──────  the project's mechanistic contribution → research-paper-friendly

## What was added

`masterarbeit_ansys_fem/extensions/bone_remodeling_huiskes.py`

  Implements the canonical Huiskes (2000) site-specific apparent-density law:

      S(x) = U(x) / ρ(x)                       SED per unit density (the "stimulus")
      dρ/dt = +B (S − k(1+w))   if S > k(1+w)  formation
              −B (k(1−w) − S)   if S < k(1−w)  resorption
              0                 else (lazy zone)
      E(ρ) = C ρ^γ                              Currey power law (γ = 2)

  Defaults: k = 4.0e-3 J/g, w = 0.35, γ = 2, C calibrated so ρ = 1.85 → E = 13700 MPa (cortical).
  Three panels in `figures/fig_bone_remodeling_huiskes.{pdf,png}`:

    (A) convergence under normal load (initial homogeneous ρ → site-specific steady state)
    (B) three loading regimes side-by-side: stress shielding (×0.30) → resorption near the crest;
        normal (×1.00) → mild densification; hyper-load (×2.50) → cortical-class densification
    (C) Huiskes vs RANKL/OPG-proxy equivalence: |Δρ|/|ρ| < 1 % over the full depth profile

  The script is pure Python (numpy + matplotlib); no Abaqus dependency. Runtime < 1 s.

## Equivalence with the RANKL/OPG disease model

The disease model (commit `cb8f18a`, `extensions/fig_periimplantitis_rankl_opg*.py`) solves a reduced
Lemaire/Pivonka ODE with explicit osteoclast (OC) / osteoblast (OB) populations + TGF-β coupling.
Under steady-state, stress-modulated dynamics, the Huiskes formation / resorption rates map onto:

    formation rate    ≈  +B (S − k(1+w))       ↔   OB activation by TGF-β released during resorption
    resorption rate   ≈  −B (k(1−w) − S)       ↔   RANKL/OPG-driven OC activation

i.e. the same SED → density curve is recovered (Panel C of the figure). This is the bridge note that
lets the thesis state both:

  > "The mechanostat threshold k separates stable osseointegration from stress-shielded resorption
  >  (panel B)"                                                                       [Huiskes language]

and

  > "The same density evolution emerges from a mechanistic RANKL/OPG remodeling ODE under TGF-β
  >  coupling (panel C, < 1 % drift), so the phenomenological Huiskes law is a steady-state limit of
  >  the disease model rather than an alternative."                                  [research language]

## Why this matters now

1. Soleimani (Erstprüfer = numerical/FEM examiner) will probe "is your bone modelled like the
   classical implant FEM literature?" — Huiskes is the universal answer.
2. The thesis can keep the mechanistic RANKL/OPG as its independent contribution, but ground it in
   the classical phenomenological law that the examiner expects.
3. Phase-field handles **structural / interface** dysbiosis → detachment; bone remodeling here
   handles **density / continuum** bone-update behaviour. Together they cover the two clinically
   reported failure modes (interface delamination + crestal bone loss / stress shielding).

## Run

    cd masterarbeit_ansys_fem/extensions
    python bone_remodeling_huiskes.py        # < 1 s, writes fig_bone_remodeling_huiskes.{pdf,png}

## What is NOT done (Outlook)

- **Coupled FEM-remodeling iteration in Abaqus.** This Python prototype computes the density update
  on a 1-D axial profile with a closed-form SED expression. A production coupled FEM run requires
  Abaqus job restart with `*MATERIAL` cards updated per region per iteration — straightforward
  scripting (see `coupling_prototype/abaqus/run_*` patterns) but ~1 day to wire in.
- **3-D density field.** The 1-D profile is sufficient for the equivalence demonstration; the 3-D
  field would be one .npz per Stage A converged solve, with the Huiskes update applied per element.
- **Time-resolved comparison.** Currently we compare *steady states*; the *transient* RANKL/OPG
  response (TGF-β release dynamics, OC turnover) is shorter than the Huiskes density update time
  scale, so the transient comparison would require care with the time-scale separation.

These are Stage-C / paper-grade extensions; the steady-state equivalence panel is sufficient for
the thesis.
