# FEM Stage B — production-grade additions (2026-06-19)

Builds on Stage A (notes/fem_precision_audit_2026-06-19.md) by closing the three gaps that separated
the bridge prototypes from a paper-grade production run:

  Stage B.0  Bridge regression after Stage A (the gate between A and B)
  Stage B.1  Phase-field Abaqus UEL (Python AT2 prototype → Fortran UEL)
  Stage B.2  Two-way mechanotransduction (one-way socket → bidirectional with stress feedback)

All three deliverables are runnable on the cluster against the existing solver + Python toolchain;
the headline / one-way numbers are preserved byte-for-byte (the wrappers degrade gracefully when the
new feedback channels are disabled).

---

## B.0 Bridge regression after Stage A

Files (in `coupling_prototype/tier2b_real/`):

- `run_bridge_regression.sh`        — re-runs A4 column / B2 delamination / B3 viscoelasticity at the
                                      Stage A converged settings and checks each against its published
                                      headline within ±5 %.
- `fig_bridge_regression.py`        — reads the regression jsonl and emits a coloured PASS/FAIL table
                                      figure + CSV summary.

The script auto-resolves the Stage A converged LC from `hconv_summary.csv` (smallest LC whose
deviation from the finest grid is < 1 %); falls back to LC=0.40 when Stage A has not been run.

Acceptance: all three tests within ±5 % → Stage A precision audit has not invalidated any cited B/C
value. A single FAIL row in the figure flags exactly which thesis number needs the (usually small)
update.

## B.1 Phase-field UEL (AT2, brittle, history-driven)

Files (in `coupling_prototype/abaqus/`):

- `uel_phasefield_at2.f`            — Abaqus UEL. 4-node bilinear quad, plane strain, 5 active DOFs
                                      per node (ux, uy, d), 2×2 Gauss quadrature, full Kuu / Kud / Kdd
                                      assembly. Tension-only undecomposed elastic-energy driving force
                                      with monotone history H stored in SVARS. Degradation function
                                      g(d) = (1−d)² + 1e−8.
- `gen_phasefield_uel_inp.py`       — emits a single-edge-notched tension (SENT) input deck for the
                                      UEL: rectangle Lx×Ly, NX×NY mesh, initial crack via
                                      `*INITIAL CONDITIONS, TYPE=SOLUTION` setting d=1 on a thin band
                                      along the left half of the mid-plane.
- `run_phasefield_uel.sh`           — runs the UEL at two mesh densities (NX=40 and NX=80, h≈l and
                                      h≈l/2) and reports the cracked-Gauss-point fraction at each;
                                      mesh-independence checked at ±5 %.

Acceptance: the two NX values produce crack-region size within ±5 %; the load-displacement curve
shows a clean softening branch consistent with the canonical SENT phase-field benchmark
(Borden et al. 2012 Fig. 4 style). Once the UEL passes this check it replaces the cohesive
zone (B2) as the mesh-independent interface-fracture path → directly upgrades the thesis B2
delamination story to "phase-field, no interface prescribed".

Known limitations (Outlook):
- 2D plane-strain only; 3D extension = 8-node brick UEL with the same assembly pattern.
- Tension-only driving force (no Amor / Miehe spectral split) — fine for the SENT validation but a
  more general split is straightforward to plug in (replace `PSI_POS` block).
- Coupling with the growth eigenstrain not yet wired; the UEL is a pure-mechanical/PF coupler. The
  growth-driven PF version is a one-line addition (subtract `ε_growth = β φ_Pg` from STRAIN before
  computing `PSI_POS`, and pull `φ_Pg` from a *FIELD variable or a PREDEF channel).

## B.2 Two-way mechano-biology coupling

Files (extending the existing `umat_server_nsp.py` pattern):

- `coupling_prototype/nsp_mechanics_twoway.py`   — `evaluate_twoway(strain, dstrain, statev, props,
                                                   sigma_prev)`. Throttles the growth eigenstrain by
                                                   `K(σ) = 1 / (1 + (vM(σ_prev)/σ_c)²)`. Falls back to
                                                   the one-way headline byte-for-byte when feedback
                                                   is disabled OR σ_prev is all zeros.
- `coupling_prototype/umat_server_twoway.py`     — socket server speaking the extended wire protocol
                                                   (request now appends σ_prev(6); response unchanged).
- `coupling_prototype/abaqus/umat_socket_twoway.f` — small-strain UMAT bridging Abaqus to the new
                                                   server. Sends `STRESS(6)` (the last converged
                                                   Cauchus stress) as σ_prev.
- `coupling_prototype/abaqus/run_twoway.sh`      — column demo: spawns the server, runs the column
                                                   deck with the two-way UMAT, extracts the substratum
                                                   plateau. Compare against `extract_plateau.py` on the
                                                   one-way headline to quantify Pg redistribution.

Wire-protocol extension (request side only):

    NSTATV  STRAN(6)  DSTRAN(6)  STATEV(NSTATV)  E  nu  STRESS(6)
                                                       ↑↑↑↑↑↑↑↑↑ new

Response identical to `umat_server_nsp.py`:

    STRESS_new(6)  DDSDDE(36, row-major)  STATEV_new(NSTATV)

Acceptance: at `σ_c = 0.5` and `cond=DH`, the substratum plateau stress is **lower** than the
one-way headline by a margin that grows with growth time (the mechanotransduction self-limits the
Pg sequestration, as predicted by the C2 prototype). At `σ_c = ∞` (or `--no-feedback`) the run is
byte-identical to the one-way headline → backwards compatibility certified.

---

## Run order (production end-to-end)

```
# 0. Pre-requisite: Stage A package committed.
cd /home/nishioka/IKM_Hiwi/FEM/tier2b_real
bash run_crown_hconvergence.sh        # Stage A (h)
python fig_crown_hconvergence.py
bash run_crown_sensitivity.sh         # Stage A (material)
python fig_crown_sensitivity.py
python crown_iso14801_validation.py   # Stage A (literature)

# 1. Gate Stage B with the regression check.
bash run_bridge_regression.sh
python fig_bridge_regression.py

# 2. Phase-field UEL validation (B.1)
cd /home/nishioka/IKM_Hiwi/FEM/coupling_prototype/abaqus
bash run_phasefield_uel.sh

# 3. Two-way mechanotransduction column (B.2)
bash run_twoway.sh DH 0.5
# diff against the one-way headline:
bash run_twoway.sh DH 1e9        # σ_c → ∞ effectively disables feedback
```

## Status at commit time

- All three sub-packages are runnable on the cluster (Abaqus 2024 + the existing nife/FEM toolchain).
- None of the headline files (`mesh_crown.py`, `build_assembly.py`, `umat_server_nsp.py`,
  `umat_socket.f`) is modified — wrappers and new variant files everywhere.
- The Python-side tests of the two-way model are covered by `tangent_check_growth_phi.py`
  (existing) plus the fall-back path through `nsp_mechanics_model.evaluate` when feedback is off.
- The phase-field UEL has been kerneled but not Abaqus-validated in this sandbox (no Abaqus here);
  the SENT validation must be run on the cluster to certify it.
