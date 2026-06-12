# Build & run on the real solvers (Abaqus now, ANSYS after seat)

The Python side runs anywhere; only the UMAT/USERMAT build needs the commercial solver + a
Fortran/C compiler. The C socket shim (`abaqus/bridge_client.c`) is shared by both solvers.

## Abaqus (your own license — prototype here first)
Prereqs: Abaqus + a compatible Fortran (Intel ifort/ifx) and C compiler; `abaqus` on PATH.

```bash
cd coupling_prototype
# placeholder (linear elastic, no JAX) — fastest sanity check:
abaqus/run_abaqus_test.sh placeholder
# NSP-backed (needs JAX in the python on PATH; deck already set to *DEPVAR 12):
abaqus/run_abaqus_test.sh nsp
```
What the script does: compiles `bridge_client.o`, starts the matching Python server, runs the
1-element job `one_element.inp` with `user=umat.f`, links the C shim, stops the server.

Checklist if it fails:
- **Linker can't find `bridge_eval_`** → name mangling. gfortran/ifort append one underscore
  (`bridge_eval_`, already used). If your ifort uses no underscore, drop it in `bridge_client.c`.
- **`abaqus make` ignores the .o** → pass it via `link_sl=` (done in the script) or add to
  `abaqus_v6.env` `link_sl`/`compile_fortran`.
- **Connection refused** → server not up yet; raise the `sleep` (JAX JIT warm-up) or check
  `BRIDGE_HOST/PORT`.
- **Job diverges at large strain** → expected with the CLSM-calibrated depth growth (eps_zz>1,
  finite strain). Use small steps for the smoke test, or the placeholder server.

## ANSYS (IKM/LUH seat — after access is arranged)
`ansys/usermat.f` is the USERMAT port; it reuses `bridge_client.c` unchanged. Same wire protocol,
same server. Differences handled in the code: USERMAT signature + the (xy,yz,xz)↔(xy,xz,yz) shear
reorder. Build as a UPF (`ANSUSERMAT`, link `bridge_client.o`, rebuild the solver) and activate with
`TB,USER` + `TB,STATE` (nStatev=12 for NSP). Match Felix's ANSYS version for ABI compatibility.

## Promotion path (once correctness is confirmed)
One TCP round-trip per Gauss point is fine for verification but slow for production. In order of
effort: persistent socket (cache fd in the shim) → embedded CPython (no socket) → offline-tabulated
surrogate of the constitutive response (fastest; reuses the NIFE Bayesian/GP surrogate tooling).
