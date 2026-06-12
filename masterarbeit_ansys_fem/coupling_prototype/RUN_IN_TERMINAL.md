# Final Abaqus confirmation — run in a real terminal

The CI sandbox SIGKILLs the persistent socket server, so the NSP/finite-strain Abaqus run must be
done in a real terminal. Everything is wired; this is one command + a glance at the output.

## Prerequisites (one-time)
- `abaqus` on PATH (Abaqus 2024 OK): `export PATH="$HOME/DassaultSystemes/SIMULIA/Commands:$PATH"`
- Python with numpy (no JAX needed at solve time — the trajectory is pre-tabulated).
- Files already present in `coupling_prototype/`: `nsp_traj_DH.json` (5-step growth ramp),
  `umat_server_fs_tab.py`, `fs_kernel.py`, `abaqus/{umat_socket_fs.f,one_element_fs.inp,extract_fs.py}`.

## A. Smoke test — bridge on real Abaqus (placeholder, ~5 s)
```bash
cd coupling_prototype
abaqus/run_abaqus_test.sh placeholder
```
Expect: `Abaqus JOB one_element COMPLETED`, σ_xx = 50.0 = E·ε, ε_lat = −ν·ε.  ← already verified here.

## B. Finite-strain NSP growth (the pending confirmation, ~30 s)
```bash
cd coupling_prototype
abaqus/run_abaqus_test.sh nsp
```
What it does: starts the numpy surrogate server, runs the NLGEOM 1-element job where the
composition-driven growth eigenstrain (Fg = diag(1,1,λ_z), λ_z ramps 1.0→1.3 over 5 increments)
deforms a substrate-bonded, free-top element, then prints the extracted result.

**PASS looks like** (`extract_fs.py` output):
```
frame  U3(top)    |S|max       SDV1
0      0.0000     ~0           0
...
5      ~0.30      low (free top) 5
```
i.e. the element **thickens in z** (U3(top) → ~λ_z−1 ≈ 0.30), stress stays **low** (free top relieves
the growth), and **SDV1 climbs 1→5** (one NSP step per converged increment). If it diverges, lower
the ramp endpoint in `nsp_traj_DH.json` or shrink the increment in `one_element_fs.inp`.

## B2. Residual-stress column — the FEM-only result (~1 min)
The scientific payload a single element cannot produce: a multi-element column with **depth-graded
growth** from the CLSM φ_Pg(z) profile, laterally confined → an in-plane **residual-stress profile**.
```bash
cd coupling_prototype
abaqus/run_abaqus_test.sh col        # generates column_fs.inp (8 elems), runs, extracts σ(z)
```
**PASS looks like** (`extract_col.py`):
```
z       elem   S11(in-plane)  S33(thru)   SDV1
0.06    1      -1700ish       ~0          20
0.44    4      -800ish        ~0          20     <- Pg-poor mid-layer: least compression
0.94    8      -2200ish       ~0          20     <- Pg-rich top: most compression
```
i.e. **S11(z) compressive, magnitude tracking φ_Pg(z)** (most at Pg-rich layers, least mid-depth),
S33 ≈ 0 (free top). That depth-varying residual stress is the differential-growth result that
motivates the FEM. The server now uses the **consistent neo-Hookean Jacobian** (robust convergence)
and supports two growth physics — edit the `col` server line in `run_abaqus_test.sh`:
`--growth_mode iso` (volumetric, default → in-plane residual stress) or `--growth_mode depth`
(anisotropic substratum-normal); `--gamma` sets magnitude (CLSM-calibratable ~17; 0.4 = easy demo).

## C. Longer / confined variants (optional)
- Longer growth: regenerate a longer ramp and match the increment count:
  ```bash
  python -c "import json,numpy as np; phi=np.linspace(0.078,0.160,40).tolist(); \
    json.dump({'cond':'DH','phi_Pg':phi},open('nsp_traj_DH.json','w'))"
  # then set *STATIC,DIRECT 0.025,1.0 and INC=200 in one_element_fs.inp
  ```
- JAX-live server instead of the surrogate: `abaqus/run_abaqus_test.sh nsp_jax`.
- Confined (no free top) to see large growth stress: remove the top-face freedom in the deck.

## Report the result
Paste the `extract_fs.py` table back; I'll fold "rung 3b VERIFIED on Abaqus" into
`REPORT_pre_ANSYS.md` and the README, then we move to the ANSYS USERMAT port (after the seat).
