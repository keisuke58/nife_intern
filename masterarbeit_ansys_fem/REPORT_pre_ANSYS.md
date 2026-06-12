# Pre-ANSYS Progress Report — Python material model × FEM

**Thesis (proposed):** *Finite-Element Implementation of a Hamilton-Principle Material Model for
Oral Biofilms — Gauss-Point Coupling of a Python Constitutive Model in ANSYS*
**Examiners:** Soleimani (1st / Erstprüfer, topic proposer) · Geisler (2nd / Zweitprüfer) · Junker (PI) · FEM codebase: Felix (IKM)
**Date:** 2026-06-12 · **Status:** prototype proven on Abaqus; ready to port to ANSYS once a seat is arranged.

---

## 1. Objective
Replace the phenomenological constitutive law in IKM's existing ANSYS FEM with a **Python material
model invoked at each Gauss point**. Prototyped in Abaqus (own license) while ANSYS access is
arranged; the design maps 1:1 onto ANSYS USERMAT.

The "material model" is the oral-biofilm **NSP (Hamilton-principle) model** from the recent work —
the same per-node reaction the spatial PDE (`nsp_pde_1d_heine.py`) already vmaps over z-nodes, here
generalised from a 1-D method-of-lines grid to 3-D element Gauss points.

## 2. What was built (prototype ladder, `coupling_prototype/`)
| Rung | Content | Status |
|------|---------|--------|
| 1 | placeholder linear-elastic + FD consistent-tangent check | ✅ PASS |
| 1b | **real Heine NSP** material model + AD-vs-FD tangent | ✅ PASS (rel-err 5e-5) |
| 1c | **gLV vs NSP** comparison (5-state vs 12-state, both FEM-ready) | ✅ PASS |
| 1d | φ_Pg → growth eigenstrain → Cauchy stress (mechanically closed) | ✅ PASS |
| 1e | **CLSM β calibration** of the growth coupling | ✅ DH corr=0.88 |
| 1f | **finite strain** F = Fe·Fg + neo-Hookean | ✅ PASS |
| 2 | Python↔Fortran socket bridge (wire protocol) | ✅ exact |
| 3 | **Abaqus UMAT (small strain) — VERIFIED on Abaqus 2024** | ✅ σ_xx=E·ε |
| 3b | **Abaqus NSP finite-strain (NLGEOM) — VERIFIED on Abaqus 2024** | ✅ U3→0.302=λ_z−1, σ=0, SDV1 0→20 |
| 3c | **SOCKET-FREE growth UMAT** (`umat_growth_phi.f`) — φ_Pg via FIELD var, geometric tangent; **FD tangent check** (`tangent_check_growth_phi.py`, Kirchhoff/Jaumann) | ✅ U3=0.30011=λ_z−1 (err 0), σ~2e-14; tangent FD rel-err **5e-8** (full CLSM magnitude) |
| 3d | **SOCKET-FREE substratum-interface column** (φ_Pg(z) field, **tall H=4×w**, no server) — **γ CLSM-calibrated** (iso volume-matched s=Jg^⅓, real DH/CH β,ic) + **mesh convergence** (A4) | ✅ stress = substratum boundary layer (free top relieves); within it σ_xx tracks φ (corr +1.0); **DH≫CH peak 6.4×, mean 10×**; converged N16 (+0.27%) |
| 4 | ANSYS USERMAT port (near-copy of 3c: FIELD→ANSYS field var) | ⏳ after ANSYS seat |

## 3. Key results
**3.1 Bridge proven on the real solver.** Abaqus 2024 (Intel ifort) ran a 1-element C3D8 with the
self-contained socket UMAT (`umat_socket.f`, TCP via Fortran ISO_C_BINDING — no external object to
link). At 1% uniaxial strain it returned σ = [50.0, 0, 0, …], ε = [0.01, −0.0045, −0.0045, …], i.e.
**σ_xx = E·ε_xx and ε_lat = −ν·ε_xx exactly** — the Python model's stress delivered through the
socket at each Gauss point. The coupling works on a commercial FE solver, not just a mock.

**3.1b Finite-strain NSP growth verified on Abaqus.** The NLGEOM 1-element job
(`run_abaqus_test.sh nsp`) completed on Abaqus 2024, 20 increments all converged. The free-top
element thickened in z by **U3(top) = 0.302 ≈ λ_z−1** (the prescribed growth stretch) at **zero
stress** (z-only growth relieved by the free surface), with **SDV1 climbing 0→20** (one NSP step per
converged increment). The composition-driven finite-strain growth (F = Fe·Fg) flows through the real
solver per Gauss point — and the stress-free thickening matches the model's free-top prediction
(§3.4). The solve-time server was the JAX-free numpy surrogate.

**3.2 Both constitutive laws are FEM-ready.** Same Heine fit (A, b), two laws: gLV replicator
(5-state, tangent ~1e-11) and NSP (12-state, ~5e-5). Both supply a consistent tangent — the
prerequisite for the implicit FE Newton loop. → "same data, two constitutive models" is a clean
thesis comparison axis.

**3.3 Growth coupling grounded in CLSM data (new finding).** Regressing biofilm thickness (CLSM
`zprofiles_all_ti.csv`) against the P. gingivalis fraction:
- **DH (dysbiotic): corr 0.88, β_depth ≈ 17.2** — thickness 22 → 79 µm as φ_Pg 0.08 → 0.21.
- **CH (commensal control): opposite sign** — Pg drives thickening only in dysbiosis.
Because the flow-chamber lateral FOV is fixed, growth is **anisotropic (substratum-normal z)** → the
eigenstrain belongs in zz, not isotropic swelling.

**3.4 The growth is finite strain (modelling consequence).** Calibrated ε_zz reaches ~1.2–2.4
(λ_z ~ 2.4). The small-strain additive split is *invalid* there; the model must use the
**multiplicative split F = Fe·Fg, F_g = diag(1,1,λ_z(φ_Pg))** with a hyperelastic (neo-Hookean) Fe.
Confined growth costs ~15–20·μ in both formulations (140% growth can't be confined cheaply), but
only F=Fe·Fg is kinematically valid; the realistic free-top BC is ~stress-free (biofilms thicken
without failure). **Meaningful residual/delamination stress needs spatial growth gradients — i.e.
the FEM job, which is the thesis.**

## 4. Architecture note — the surrogate path
The solve-time server was made **JAX-free** (`umat_server_fs_tab.py` + `fs_kernel.py`): the NSP
composition trajectory φ_Pg(step) is precomputed offline with JAX, and the per-Gauss-point server is
pure-numpy neo-Hookean. This is the documented production-promotion path (offline-tabulated
surrogate) and keeps the constitutive server lightweight — reusing the NIFE Bayesian/GP surrogate
tooling later.

## 5. Open items before ANSYS
1. **ANSYS seat** via IKM/LUH campus license — confirm with Felix/Soleimani (route A: account on the
   machine where Felix's model already runs; match his ANSYS version for ABI). *Blocking for rung 4.*
2. ~~Finite-strain NSP live run on Abaqus~~ — ✅ **DONE** (verified on Abaqus 2024, §3.1b).
3. ~~φ→stress coupling physics~~ — ✅ **DONE**: growth modes `iso` (volumetric → in-plane residual
   stress) and `depth` (anisotropic substratum-normal) selectable; γ documented as CLSM-calibratable
   (~17 for full thickness match; demo uses a convergence-friendly value). Further: μ from biofilm
   rheology, buckling/delamination (multi-element, follow-on).
4. ~~Production tangent~~ — ✅ **DONE**: consistent neo-Hookean Jaumann Jacobian (`material_jacobian_nh`,
   λ'=λ/J, μ'=(μ−λlnJ)/J) replaces the approximate elastic one in both FS servers; reduces exactly to
   the elastic tangent at Fe=I.
5. **Handover meeting with Felix** + acknowledge his PhD FEM contribution.

## 6. Administrative
- Registration form downloaded (`Zulassung_MA_fillable.pdf`, restriction-free, 2 pages).
- Reply to Soleimani drafted (`notes/reply_to_soleimani_DRAFT.md`) — includes title alignment with
  Junker, ANSYS-license question, Abaqus-prototyping note, milestone planning around the NIFE internship.
- **Open decision:** Junker + Soleimani must agree on ONE title before Page 1 goes to the Prüfungsamt.

## 7. Bottom line
The Python↔FE coupling is **proven on a real solver**, the NSP/gLV constitutive laws are
**FEM-ready**, and the growth coupling is **calibrated on real CLSM data** with a defensible
finite-strain finding. The remaining work is (a) the ANSYS seat + USERMAT port, (b) finishing the
finite-strain live run, (c) refining the growth physics — all ready to start.
