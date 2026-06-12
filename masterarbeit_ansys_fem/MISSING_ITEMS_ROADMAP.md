# "Still-missing" items → concrete tasks (push to thesis-ready)

Owner key: **[me]** I can build/draft now · **[run]** needs your Abaqus terminal · **[human]** needs
Junker/Soleimani/Felix.

## ① Topic / framework agreement  — **[human]** BLOCKING
Decide & get written agreement: thesis FEM = **Abaqus** (you + Muramatsu/Keio use it) or **ANSYS**
(Felix's framework). Affects the title on Page 1. → email to Junker + Soleimani (offer Abaqus-now /
ANSYS-port plan). *Until this is set, ④⑤ scope and the Page-1 title are provisional.*

## ② Verification — **[run]** harness built
- Single-point consistent-tangent check ✅ (`nsp_tangent_check.py`).
- **Mesh convergence** ✅ RAN: peak interior |S11| converges (354→316→320; **N=16 vs 32 = 1.3%** →
  use N=16). S11_top is a **free-surface boundary layer** (−367→−25 as N grows: thinner top slice
  samples the Saint-Venant relaxation zone) — not a convergence metric; report interior peak.
  Figure `coupling_prototype/fig_mesh_convergence.{pdf,png}` = F5. *Thesis nuance: residual stress
  peaks in the interior and relaxes at the free surface.*
- Patch test / free-swelling exactness: the 1-element U3→λ_z−1 result already serves (rung 3b).

## ③ Validation — **[me]** framing now, data later
- Be explicit in the thesis: the residual-stress **shape** is calibrated (φ_Pg(z)); the **magnitude**
  (γ) is parametric and **not yet validated against a mechanical measurement**.
- What would validate: biofilm AFM/rheometry stress, CLSM-tracked deformation under flow, or
  detachment-onset shear vs Pg load. List as "validation pathway" (honest, examiner-proof).

## ④ Constitutive justification — **[me]** scaffold (verify cites)
Replace the generic neo-Hookean μ with biofilm-grounded values/justification:
- Oral/biofilm **shear modulus G ~ 10–1000 Pa**, highly strain- and species-dependent; biofilms are
  **viscoelastic** (creep, stress relaxation), often modelled as Maxwell / Burgers / power-law fluids
  on long timescales, ~elastic on short. *(verify: Stoodley, Klapper, Wilking/Brenner; numbers are
  order-of-magnitude established.)*
- Growth >> elastic strain timescale → the elastic response is the fast/instantaneous part; a
  compressible neo-Hookean for Fe is a defensible first model, with poro-/visco-elasticity as
  Outlook. State this explicitly.
- Use a realistic G (e.g. ~100 Pa) for any quantitative claim; report stress as σ/μ to stay
  modulus-agnostic where μ is uncertain.

## ⑤/⑥ Clinical prediction — ✅ VERIFIED in-model (DH vs CH)
Ran both conditions (Abaqus): the dysbiotic column carries **higher and far more stratified**
residual stress than commensal:

| | DH (dysbiotic) | CH (commensal) |
|---|---|---|
| peak \|S11\| | **354** | 234 |
| depth amplitude (max−min) | **218** | 81 |
| profile | strongly stratified (peak at Pg-rich z≈0.87) | nearly uniform (~−220) |

→ **DH ≈1.5× peak, ≈2.7× depth-heterogeneity** ⇒ mechanically more prone to interfacial
failure/detachment; commensal is mechanically uniform/benign. Figure
`coupling_prototype/fig_residual_stress_DHvsCH.{pdf,png}` (driver φ_Pg(z) → result S11(z)). This is
the testable clinical claim, now supported. *(Caveat ③: shape calibrated, γ-magnitude parametric.)*

## ⑤ Clinical interpretation — **[me]** draft narrative
- Differential growth (Pg-rich layers grow more) under substrate adhesion → **in-plane compressive
  residual stress** that peaks where Pg accumulates (substratum + maturing surface) and dips in the
  Pg-poor mid-layer (the F4 profile).
- Mechanism → consequences: residual stress drives **interfacial shear at the implant surface →
  micro-delamination / detachment**, seeding dispersal and deepening peri-implant pockets; couples
  to the clinical dysbiosis story already in ch1/ch5 "Clinical Applications".
- Testable prediction: dysbiotic (DH) columns carry higher peak residual stress than commensal (CH)
  → higher detachment propensity. → run CS/CH/DS/DH profiles (⑥) to support it.

## ⑥ Scope / multi-condition + writing — **[run]+[me]**
- Per-condition residual-stress profiles: build `phipg_depth_{CS,CH,DS,DH}.json` **[me]**, run each
  **[run]**, compare peaks (supports ⑤'s prediction).
- 2-D/3-D realistic geometry: Outlook unless time permits.
- Writing: the new ch5 section (see `INTEGRATION_PLAN.md`) — draft on approval (thesis-freeze).

## Suggested order (fastest value)
1. **[human]** send the topic/seat email (unblocks everything). 
2. **[run]** `run_convergence.sh` → verification figure.
3. **[me]** per-condition profiles + plotting scripts; ④/⑤ draft text.
4. **[run]** per-condition column runs → ⑤'s DH>CH prediction.
5. **[me]** draft the ch5 section (on your go) and the figures.
