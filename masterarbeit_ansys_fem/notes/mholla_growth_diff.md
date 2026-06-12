# mholla/growth ↔ our coupling_prototype — diff & portable parts (2026-06-13)

Repo: https://github.com/mholla/growth (Maria Holland, *Hitchhiker's Guide to Abaqus*,
Zenodo 1243270 — **must attribute if we reuse**). Self-contained Fortran growth UMATs (no socket).
Cloned to `/tmp/mholla_growth`.

## What matches us exactly
Our `fs_kernel.py` and mholla use the **identical neo-Hookean** Cauchy stress:
- mholla `umat_iso_morph.f:117`: `σ_i = ((λ·lnJe − μ)·ξ_i + μ·be_i)/Je`
- ours `cauchy_neo_hookean()`: `σ = (μ/J)(b−I) + (λ/J)lnJ·I`  → algebraically the same.

Growth-split correspondence (Fe = F·Fg⁻¹):
| our `fs_kernel.py`            | mholla file            | growth tensor Fg                  |
|------------------------------|------------------------|-----------------------------------|
| `growth_gradient_iso`        | `umat_iso_morph.f`     | θ^(1/3)·I  (isotropic volumetric) |
| `growth_gradient_depth` (λ_z)| **`umat_fiber_morph.f`** | I+(θ−1)·n⊗n, **n=e_z ⇒ diag(1,1,θ)** |

→ `umat_fiber_morph.f` with `xf0 = (0,0,1)` is **literally our anisotropic substratum-normal depth
growth**. This is the cleanest Fortran template for the ANSYS USERMAT / socket-free Abaqus port.

## THE portable win — geometric tangent terms we are missing
Our `material_jacobian_nh()` (fs_kernel.py:35) returns **only the elastic moduli** (λ', μ' block):
```
C[:3,:3]=λ', diag+=2μ', shear=μ'      # elastic part ONLY
```
mholla's `ddsdde` (umat_iso_morph.f:121-144) adds the **finite-strain geometric / co-rotational
stress terms** that Abaqus' Jaumann-rate convention requires:
```
ddsdde(1,1) = (λ − 2(λlnJe−μ))/Je + 2·σ_1     ! note the +2σ geometric term
ddsdde(1,4) = σ_4 ;  ddsdde(4,4) = −(λlnJe−μ)/Je + (σ_1+σ_2)/2 ;  ddsdde(4,5)=σ_6/2 ...
```
Our server's approximate tangent omits the `+2σ`, `σ_4`, `(σ_i+σ_j)/2` contributions. **Consequence:
worse Newton convergence under NLGEOM** (we currently lean on small increments / 20 steps). Porting
these terms = fewer iterations, larger increments, and an examiner-defensible "consistent tangent".

## ✅ DONE (2026-06-13) — socket-free UMAT built & verified on Abaqus 2024
`coupling_prototype/abaqus/umat_growth_phi.f` (rung 3c) realizes extraction items 1+2:
- **φ_Pg as a PREDEFINED FIELD variable** (PREDEF(1)) — no socket, no Python. The offline-surrogate
  path and the exact pattern an ANSYS USERMAT uses (field var → growth stretch).
- Depth growth `Fg=diag(1,1,λ_z)`, `λ_z = 1+max(0, β·φ_Pg+ic)` with the **CLSM DH calibration**
  (β=17.1806, ic=−1.3372 from `beta_calibration.json`); `mode=1` switch for isotropic Fg.
- **mholla geometric consistent tangent** ported in (the `+2σ`, `σ_4`, `(σ_i+σ_j)/2` terms).
- Verified, two decks (`one_element_phi.inp` free / `one_element_phi_confined.inp` confined):
  free-swelling **U3=0.30011=λ_z−1 (err 0), σ~2e-14**; confined **S33=−0.69μ** compressive, both
  converging in **1–2 Newton iterations/increment** ⇒ the consistent (geometric) tangent works.
- Reproduces the socketed rung 3b (U3=0.302, σ=0) **socket-free** ⇒ ready for the ANSYS USERMAT
  port (item 4): swap PREDEF→ANSYS field var, same stress+tangent math.

## Extraction plan (concrete) — original, now mostly realized above
1. **[surrogate UMAT, socket-free]** Take `umat_fiber_morph.f`, hard-set `xf0=(0,0,1)`, and replace
   its time-linear growth `theg = 1 + alpha*time(2)` with our **φ_Pg-driven** stretch
   `theg = 1 + max(0, β_depth·φ_Pg + intercept)` (β_depth, intercept from `beta_calibration.json`).
   φ_Pg(step) is read from a precomputed table (our offline-surrogate path) via a PROPS index or a
   STATEV seeded by `sdvini`. Result: a pure-Fortran depth-growth UMAT with **no Python server** —
   the production/ANSYS form. Keep `umat_socket_fs.f` as the JAX-faithful reference for verification.
2. **[tangent upgrade]** Port mholla's geometric `ddsdde` terms into `material_jacobian_nh()` (and the
   FS server) so the socketed path also gets the consistent tangent. Cross-check against mholla's
   Fortran on the 1-element `cube_1_C3D8_stretch_*` cases.
3. **[verification freebies]** mholla ships matching `input_files/*.inp` (1- and 8-element cubes,
   stretch in x/y/z, no-load `muffin_noload.inp` for iso growth). Reuse these as **independent
   verification** of our stress/tangent (same material, two implementations ⇒ machine agreement is a
   clean A-block verification item).

## Attribution / license
Repo is academic-share; README requires citing the **Hitchhiker's Guide to Abaqus** (Zenodo
record/1243270). If any `ddsdde`/Fe code is lifted, cite it in the thesis methods + acknowledge.
Our socket architecture and the φ_Pg/Hamilton coupling are original; mholla supplies the textbook
neo-Hookean+growth Fortran kernel and the geometric tangent.
