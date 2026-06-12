# Python ↔ FEM coupling prototype (Abaqus sandbox → ANSYS target)

Goal: invoke a **Python material model at each Gauss point** of an existing FEM solver, replacing
the phenomenological constitutive law. Prototyped in Abaqus (own license) while ANSYS access is
arranged; the design maps 1:1 onto ANSYS **USERMAT**.

Two material models live here:
- `material_model.py` — **placeholder** (isotropic linear elasticity), drives the socket/UMAT
  plumbing demo (rungs 2–3) with no heavy deps.
- `nsp_material_model.py` — the **real Heine 2025 NSP (Hamilton) model**, wrapping the canonical
  `_newton_step` from `Tmcmc202601/data_5species/main/hamilton_ode_jax_nsp.py` (JAX). This is the
  same per-node reaction `nsp_pde_1d_heine.py` vmaps over z-nodes — i.e. the spatial PDE already
  *is* "invoke the model at each node", which FEM generalises to Gauss points. STATEV = the 12-dim
  NSP state `g = [phi_1..5, phi_0, psi_1..5, gamma]`.

> Physical-role caveat: NSP returns a **composition/reaction** update, not a mechanical stress.
> At a Gauss point it supplies a source / internal-variable evolution; the real thesis modelling
> step is the `phi -> stress` (growth/swelling) coupling that closes the mechanics.

## Prototyping ladder
| Rung | File | What it de-risks | Status |
|------|------|------------------|--------|
| 1. Single-point driver | `single_point_driver.py` | placeholder law: sane stress + **consistent tangent** (FD-checked) | ✅ runs, PASS |
| 1b. Real NSP model | `nsp_material_model.py` + `nsp_tangent_check.py` | real Heine NSP step + its AD tangent vs FD at a Gauss point | ✅ runs, PASS (rel-err 5e-5) |
| 1c. gLV vs NSP | `glv_material_model.py` + `compare_glv_nsp.py` | comparison baseline: gLV replicator (5-state) vs NSP (12-state), both tangent-checked | ✅ PASS (gLV 1e-11, NSP 5e-5) |
| 1d. Closed mechanics | `nsp_mechanics_model.py` + `nsp_mechanics_driver.py` | phi_Pg → growth eigenstrain → **Cauchy stress**; confined growth stress | ✅ PASS |
| 1e. CLSM β calibration | `calibrate_beta.py` → `beta_calibration.json` | ground the growth coupling in real CLSM thickness data | ✅ DH corr=0.88, β_depth=17.2 |
| 1f. Finite strain | `nsp_mechanics_fs.py` + `nsp_mechanics_fs_driver.py` | F=Fe·Fg multiplicative growth + neo-Hookean (λ_z~2.4 is finite strain) | ✅ confined ~−18μ, free-top ~0 |
| 2. Socket bridge | `umat_server.py` + `umat_client_test.py` | the Python↔Fortran **wire protocol** is exact, no Abaqus needed | ✅ runs, PASS (err 0) |
| 3. Abaqus UMAT (small strain) | `abaqus/umat_socket.f` + `one_element.inp` | UMAT calls the server per Gauss point on a 1-element job | ✅ **VERIFIED on Abaqus 2024**: σ_xx=50.0=E·ε, ε_yy=ε_zz=−νε ✓ |
| 3b. Abaqus NSP finite-strain | `umat_socket_fs.f` + `umat_server_fs_tab.py` + `one_element_fs.inp` | NLGEOM growth via DFGRD1; numpy surrogate server | ✅ **VERIFIED on Abaqus 2024**: 20 increments, U3(top)→0.302=λ_z−1, σ=0 (free-top), SDV1 0→20 |
| 3c. Residual-stress column | `umat_socket_col.f` + `umat_server_col.py` + `gen_column_inp.py` | multi-element, depth-graded growth from CLSM φ_Pg(z), laterally confined → σ_xx(z) | ✅ **VERIFIED on Abaqus 2024**: S11(z) compressive −136…−354, tracks φ_Pg(z); S33≈0; the FEM-only residual-stress profile |
| 4. ANSYS USERMAT | `ansys/usermat.f` (reuses `bridge_client.c`) | same bridge inside Felix's ANSYS model; handles shear-order remap | ⏳ skeleton — after ANSYS seat (see `BUILD.md`) |

## Run rungs 1–2 now
```bash
cd coupling_prototype
python single_point_driver.py      # placeholder tangent check, 3 strain paths -> PASS
python umat_client_test.py         # spawns umat_server.py, checks protocol -> PASS
python nsp_tangent_check.py --cond DH   # REAL Heine NSP model, AD-vs-FD tangent -> PASS (needs JAX)
python compare_glv_nsp.py --cond DH     # gLV vs NSP, both tangent-checked -> PASS
python calibrate_beta.py                # ground beta in CLSM thickness data -> beta_calibration.json
python nsp_mechanics_driver.py --cond DH  # phi_Pg -> depth growth -> growth stress -> PASS
python umat_client_nsp_test.py          # NSP-backed bridge, STATEV(12) round-trip -> PASS
```

## Growth coupling — calibrated, anisotropic, finite-strain
`calibrate_beta.py` regresses biofilm thickness (CLSM `zprofiles_all_ti.csv`) against the Pg
fraction. **DH: depth-growth strain vs phi_Pg has corr 0.88, β_depth≈17.2** (thickness 22→79 µm);
the commensal control CH shows the opposite sign — Pg drives thickening only in dysbiosis. Because
the flow-chamber lateral FOV is fixed, growth is **anisotropic (substratum-normal z)**, so
`nsp_mechanics_model.py` defaults to `growth_mode="depth"` (eigenstrain in zz only). The calibrated
ε_zz reaches ~1.2–2.3 → **finite strain**: the small-strain elastic stress is a linearisation; the
thesis should use a multiplicative split F = Fₑ·F_g, F_g = diag(1,1,λ_z(phi_Pg)). `--mode isotropic`
keeps the volumetric-swelling variant for comparison.

## NSP-backed bridge (rung 3, real model)
`umat_server_nsp.py` is a drop-in for `umat_server.py` using the mechanically-closed NSP model and
the **same wire protocol**, so `abaqus/umat.f` is unchanged — only set `*DEPVAR 12` (done in
`one_element.inp`) and launch the NSP server:
```bash
python umat_server_nsp.py --port 50007 --cond DH --beta 0.05 &
abaqus job=one_element user=abaqus/umat.f interactive    # link bridge_client.o
```
STATEV carries the 12-dim NSP state between increments (Abaqus *DEPVAR ↔ server round-trip, verified
exact by `umat_client_nsp_test.py`).

## Rung 3 on the Abaqus machine — VERIFIED
Two UMAT variants:
- **`abaqus/umat_socket.f`** (recommended, used by the runner): self-contained — does the TCP call
  via Fortran `ISO_C_BINDING` to libc sockets, so there is **no external object to link** (avoids
  the `link_sl`-with-`analysis` error). Server host/port hardcoded 127.0.0.1:50007.
- `abaqus/umat.f` + `bridge_client.c`: the C-shim variant (kept for reference / the ANSYS port).

```bash
cd coupling_prototype
abaqus/run_abaqus_test.sh placeholder    # starts server, runs 1-element job, stops server
```
**Verified run (Abaqus 2024, Intel ifort):** 1% uniaxial strain on a C3D8 returned
σ = [50.0, 0, 0, …] and ε = [0.01, −0.0045, −0.0045, …] — i.e. σ_xx = E·ε_xx and lateral
ε = −ν·ε_xx, exactly the placeholder linear-elastic model, delivered through the socket at the
Gauss point. The Python↔Abaqus coupling works on the real solver.

### Finite-strain NSP on Abaqus (`run_abaqus_test.sh nsp`) — VERIFIED
**Completed on Abaqus 2024** (real terminal): NLGEOM 1-element job, 20 increments all converged.
The free-top element thickened in z by **U3(top) = 0.302 ≈ λ_z−1** (the prescribed growth stretch),
at **zero stress** (free surface relieves z-only growth), with **SDV1 climbing 0→20** (one NSP step
per converged increment). This confirms the composition-driven finite-strain growth flows through
the real solver per Gauss point. Pieces:
- `abaqus/umat_socket_fs.f` — NLGEOM UMAT, sends the deformation gradient DFGRD1(9) (not small
  strain), `*DEPVAR 1` step counter.
- `abaqus/one_element_fs.inp` — single C3D8, NLGEOM, no external load: the composition-driven
  growth eigenstrain deforms a substrate-bonded, free-top element (it thickens in z).
- **`umat_server_fs_tab.py` (numpy-only, NO JAX)** — the *surrogate path*: the NSP φ_Pg trajectory
  is precomputed offline with JAX (`nsp_traj_DH.json`) and the solve-time server is pure-numpy
  neo-Hookean via `fs_kernel.py`. Verified locally to return the correct F=Fe·Fg stress (F=I →
  σ_zz growth stress; step counter advances per converged increment).
- `umat_server_nsp_fs.py` — the JAX-live variant (`run_abaqus_test.sh nsp_jax`), same protocol.

To run in a real terminal:
```bash
# offline (JAX): precompute the composition trajectory
python -c "import json,numpy as np,nsp_material_model as nm; \
  s,_=nm.make_evaluator(nm.load_params('DH')); g=nm.make_initial_state(np.array([.3,.2,.2,.15,.15])); \
  json.dump({'cond':'DH','phi_Pg':[float(np.asarray(g:=s(g))[4]) for _ in range(40)]}, open('nsp_traj_DH.json','w'))"
abaqus/run_abaqus_test.sh nsp            # numpy surrogate server + NLGEOM job
```

## Wire protocol (server ↔ UMAT)
```
request : NSTATV  STRAN(6)  DSTRAN(6)  STATEV(NSTATV)  E  nu      (whitespace ASCII)
response: STRESS(6)  DDSDDE(36 row-major)  STATEV_new(NSTATV)
```
Voigt order (11,22,33,12,13,23), **engineering** shear — Abaqus UMAT and ANSYS USERMAT share this,
so rung 3→4 is a near-copy.

## Abaqus UMAT → ANSYS USERMAT mapping
| Abaqus UMAT | ANSYS USERMAT | note |
|-------------|---------------|------|
| `STRESS` | `stress` | Cauchy, same Voigt order |
| `DDSDDE` | `dsdePl` | consistent tangent 6×6 |
| `STATEV` / `*DEPVAR` | `ustatev` / `TB,STATE` | state vars |
| `PROPS` / `*USER MATERIAL` | `prop` / `TB,USER` | material constants |
| `DSTRAN`,`STRAN` | `dStrain`,`Strain` | engineering shear both |
| build: `abaqus ... user=umat.f` | `ANSUSERMAT` + `/UPF` | same C socket shim reused |

## Known limitation (call out in the thesis)
One TCP connection per Gauss-point eval is fine for **correctness** but slow for production. Options
once the model is trusted: persistent socket (cache fd), in-process embedded CPython, or an
**offline-tabulated surrogate** of the constitutive response (fastest; pairs well with the
Bayesian/GP surrogate skills from the NIFE work).
