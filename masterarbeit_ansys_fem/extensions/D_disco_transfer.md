# D — Transfer of the biofilm growth→stress→delamination machinery to semiconductor processing

**Claim for interviews (DISCO / wafer-process AI):** the FEM machinery built for this thesis —
*composition-driven eigenstrain → residual stress → interface delamination*, with verification + a
Python material model called per Gauss point — is **structurally identical** to thin-film stress and
dicing/packaging delamination in semiconductor manufacturing. The physics objects map one-to-one; only
the names change. Almost no new compute is needed — this is a framing/repackaging of verified work.

## One-to-one mapping

| Biofilm thesis object | Semiconductor / wafer-process analogue |
|---|---|
| Growth eigenstrain `F_g = diag(1,1,λ_z(φ_Pg))` | Deposition / thermal / phase-change eigenstrain in a thin film (CTE mismatch, intrinsic stress, SiC laser-modified HAZ) |
| Composition field `φ_Pg(z)` drives growth | Dopant / temperature / damage field drives the eigenstrain (e.g. laser fluence → HAZ volume change) |
| Multiplicative split `F = F_e·F_g` | Standard inelastic-strain split for film stress (Stoney → finite-strain FE) |
| Substratum-boundary-layer residual stress (DH≫CH) | Film–substrate interfacial residual stress at the die edge / street |
| Cohesive-zone delamination (DH 83% vs CH 25%) | Die-attach / passivation / underfill delamination; chipping at the dicing street |
| Bilayer growth-wrinkling (stiff crust on soft base) | Buckling/blistering of compressively stressed films on compliant underlayers |
| Patient-specific risk curve (φ_Pg → detachment) | Process-window map: parameter (fluence, feed rate) → delamination/chipping risk |
| CLSM-calibrated growth magnitude γ | Process-calibrated eigenstrain from metrology (Raman stress, profilometry) |
| Socket / predefined-field UMAT (Python model per Gauss point) | Same coupling pattern for a learned/ML process material model in ANSYS/Abaqus |

## What transfers with zero new physics
- The **verification suite** (patch test, energy conservation, MMS p≈2, mesh convergence) is
  process-agnostic — it certifies the FE coupling for *any* eigenstrain material model.
- The **cohesive-delamination + strength-sweep** workflow is exactly a die-attach reliability study.
- The **inverse-design / ML-surrogate** layer (C1/B3) is a process-window optimiser:
  "what parameter profile minimises delamination" is the same LP/surrogate problem as the probiotic
  steering, with `φ_Pg(z)` replaced by `fluence(x)` or `temperature(t)`.

## What would need new work (honest)
- Real semiconductor material data (film moduli, CTE, interfacial fracture energies) — swap the
  biofilm constitutive constants; structure unchanged.
- 3-D wafer/street geometry decks (the meshing is new; the solver path is identical).
- Coupling to a process simulator (the SiC laser-HAZ surrogate in `wafer-proc-sim/sic/`,
  see [[project_sic_laser_sim]] and [[project_nife_disco_bridge]]).

**Bottom line:** the thesis is a worked, verified instance of *field-driven eigenstrain → interfacial
failure → risk map → inverse design*. That pipeline is the core of physics-based yield/reliability
modelling in wafer processing. The transfer is a relabelling, not a rebuild.
