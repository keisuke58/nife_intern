# Integrating the FEM work into the ~80 p thesis

The existing thesis (`~/LUHsummer26/30_Masterarbeit`, biofilm-dysbiosis, Junker) already has the
perfect hooks — the FEM work is a **continuation, not a bolt-on**:

| Existing | The FEM work continues it |
|---|---|
| ch2 §"Extended Hamilton Principle for **Biofilm Mechanics**" (theory) | provides the constitutive/variational basis → FEM is its computational realization |
| ch5 §"Spatial Extension: Reaction–Diffusion PDE" + "3-D FISH depth structure" | composition varies with depth (where the species are) → FEM gives the **mechanical state** that depth structure creates |
| Title "… Spatial **Stratification**" | mechanical stratification (residual stress vs depth) literally completes the title |

> **Thesis-freeze respected.** This is a PLAN only. Draft sections live here / in `notes/`; nothing
> in `30_Masterarbeit` is edited until you approve insertion.

## Where it goes
A new **section in ch5** right after "Spatial Extension: Reaction–Diffusion PDE" and before
"Clinical Applications" (so the narrative is: composition depth-structure → mechanics → clinic).
If it grows past ~12 p, promote to its own short chapter **"ch5b: Finite-Element Realization"**.

Working section title:
> **Mechanical Stratification: Finite-Element Realization of the Hamilton-Principle Biofilm Material Model**

## Section outline (+~12–18 p, illustrative ↔ verified flagged)
1. **Motivation & coupling concept** (~2 p) — invoke the Python biofilm material model at each Gauss
   point, replacing the phenomenological law; bridge to ch2's Hamilton mechanics. *(done)*
2. **Method: Gauss-point coupling** (~3 p) — socket/UMAT architecture, self-contained ISO_C_BINDING
   UMAT, the offline-tabulated surrogate (JAX precompute → numpy solve-time). Multiplicative growth
   split F = Fe·Fg; consistent neo-Hookean Jaumann tangent. *(done)*
3. **Growth kinematics from data** (~3 p) — CLSM thickness vs φ_Pg, DH corr 0.88, β_depth≈17.2,
   commensal control opposite sign; finite strain (λ_z~2.4). Fig: thickness↔φ_Pg scatter + fit. *(done)*
4. **Verification** (~3 p) — single-point consistent-tangent check; **mesh convergence** of the
   growth column (N=4,8,16,32); patch test / free-swelling exactness. *(② harness now built — needs runs)*
5. **Results: mechanical stratification** (~3 p) — the residual-stress column: S11(z) tracks φ_Pg(z)
   (−136 mid → −354 Pg-rich), S33≈0 free-top; iso vs depth growth modes; per-condition (CS/CH/DS/DH).
   *(headline done; **real-FOV map done** — `extensions/c3_real_fov_stress.py` drives the verified
   boundary-layer model with the REAL measured φ_Pg(z) over all 84 HOBIC FOVs → detachment-risk
   distribution CH vs DH, fig `figures/F17_real_fov_stress.pdf`.)*
6. **Validation & limits** (~2 p) — honest: γ-magnitude is CLSM-calibratable but the **stress
   prediction is not yet validated against mechanical measurement**; what data would validate. *(③)*
7. **Clinical reading** (~2 p) — residual stress → detachment / delamination → peri-implantitis;
   feeds the existing "Clinical Applications". *(⑤ draft)*

## Verification/validation reuse already in the repo
- `nsp_tangent_check.py`, `compare_glv_nsp.py` → §4 single-point verification.
- `calibrate_beta.py` + `beta_calibration.json` → §3 figure.
- Abaqus-verified runs (rungs 3/3b/3c) → §2/§5 (real-solver evidence).

## Figures to produce (thesis_style.py, usetex)
- F1 coupling architecture schematic (TikZ).
- **F1b data-provenance bridge** (`gen_bridge_fig.py` → `figures/F1b_data_bridge.pdf`) — *done.*
  Patient 16S (composition, 0D, no depth) → in-vitro HOBIC φ_Pg(z) depth shape → FEM S11(z).
  Makes the **layer separation honest**: the depth shape is in-vitro CLSM, scaled by patient load.
- F2 thickness↔φ_Pg calibration scatter + fit (corr 0.88). *(done)*
- F3 finite-strain free-swelling: U3(top) vs increment = λ_z−1 (1-element verification). *(done)*
- F4 **residual-stress profile S11(z) vs depth**, overlaid with φ_Pg(z) — the headline. *(done)*
- F5 mesh-convergence (peak |S11| vs element count) — §4. *(done)*
- F17 real-FOV detachment-risk distribution CH vs DH (84 HOBIC FOVs) — §5/§7. *(done)*

## Data provenance & layer separation (state explicitly in §6)
Two distinct real datasets feed the framework at **different layers** — do not blur them:
- **Patient 16S (Dieckow peri-implant, joshi)** = real but **composition-only (0D)**; drives the
  ecological gLV/Hamilton model, *not* the FEM directly (no depth axis).
- **In-vitro HOBIC CLSM/FISH** = the **depth shape** φ_Pg(z) that drives the FEM. The Pg channel uses
  the **corrected 4ch→5sp decode** (`fish_decode.py`: Fn=blue∩red, So=blue−Fn, **Pg=red−Fn**); the
  source `zprofiles_all_ti.csv` was generated *after* that fix (commit 6d6d343, decode landed 7fe4072)
  — verified 2026-06-13, the earlier naive 1:1 channel map is *not* in this data.
- **Bridge assumption** (Fig F1b): patient supplies the Pg *load*; in-vitro CLSM supplies the *depth
  shape*; their product is the assumed φ_Pg(z). This is an assumption, not a measurement on the patient.

## Honest scoping (matches feedback_thesis_selective_inclusion)
- **Body**: method + Abaqus-verified results + CLSM calibration + residual-stress profile (real).
- **Outlook/Appendix**: ANSYS port (after seat), buckling/delamination, biofilm-specific
  viscoelastic/poroelastic constitutive law, validation against mechanical data.
- Do NOT over-claim the residual-stress magnitude as predictive until validated — present as the
  mechanism the framework produces, calibrated in shape, parameteric in magnitude.

## Page math
80 p now + ~12–18 p (this section) → ~95–100 p. Comfortable for an IKM Masterarbeit; trim the
spatial-PDE appendix if needed.
