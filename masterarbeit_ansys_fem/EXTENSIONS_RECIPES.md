# Verification & mechanics extensions — "try all" (A + B + C)

Status: **[run]** built & runnable in your terminal · **[recipe]** concrete build steps (iteration
expected; not verified here — Abaqus is absent in the dev sandbox). Soleimani (now Erstprüfer) =
numerical/FEM, so **A is the highest-value block.**

## A. Numerical verification (examiner-proof)
- **A1 Patch test — ✅ PASS (Abaqus).** `phipg_flat.json` (uniform φ_Pg=0.20): all 7 interior
  elements gave **identical S11 = −226.57** (machine-identical); the top element (−178.56) is the
  documented free-surface boundary layer. Uniform growth → uniform interior stress ⇒ the material
  model + coupling are FE-consistent. `abaqus/run_abaqus_test.sh patch`.
- **A2 Energy conservation — ✅ PASS (Abaqus).** Note: the *eigenstrain* (growth) does internal work
  that Abaqus does **not** count in ALLWK, and the socket UMAT does not fill SSE, so a naive ALLSE=ALLWK
  test is meaningless here. Clean check instead (`energy` mode, growth OFF via `phipg_zero.json`, unit
  cube pulled to u_z=0.20): the external work **ALLWK = 89.51** equals the **analytic compressible
  neo-Hookean stored energy U = 89.52** recomputed in closed form from the final stretches
  (λ_x=λ_y=0.9206, λ_z=1.20, J=1.017) — **rel-diff 1.3×10⁻⁴**, i.e. energy conserved, no spurious
  dissipation, and symmetric Poisson response. `extract_energy2.py energy_test`. (J=det(Fe) consistency
  in the growth runs is exact by construction of the multiplicative split.)
- **A3 MMS (manufactured solution) — ✅ PASS (Abaqus), the rigorous order check.** Manufactured exact
  solution `u_x = α sin(πX/L)`, `u_y=u_z=0`; the matching body force `B_x = (λ+2μ)α(π/L)²sin(πX/L)` is
  applied per element column via `*DLOAD, BX` (`gen_mms_inp.py`), growth off. Refining Nx=4→8→16→32 and
  measuring `‖u_h−u_exact‖_L2` (`extract_mms.py`) gives **observed order p = 1.98, 1.97, 1.90 (pairwise),
  least-squares 1.95 ≈ 2** — textbook trilinear-C3D8 second-order L2 convergence on a non-trivial
  (non-patch) solution. (α=1e-4 keeps the neo-Hookean O(α²) nonlinearity below the discretization error;
  at α=1e-3 the finest mesh flattens to p≈0.5 as the forcing floor is reached — a documented, expected
  artefact, not an FE bug.) Driver: `abaqus/run_mms.sh` → `mms_convergence.txt`.
- **A4 SOCKET-FREE column mesh convergence — ✅ PASS (Abaqus 2024).** The field-driven UMAT
  (`umat_growth_phi.f`) column at N=4,8,16,32. **Uniform φ_Pg=0.20**: the confined-bottom plateau
  `|σ_xx| = 0.848μ` is **machine-identical across all N** (patch/plateau verification of the element +
  growth coupling). **Graded φ_Pg(z), DH**: the confined-region stress converges to **N=16 vs 32 =
  +0.27 %** ⇒ **use N=16**. Driver: `gen_column_phi_inp.py` (+`GSCALE` moderate-growth regime,
  `HEIGHT` aspect ratio) → `extract_plateau.py`; figure
  `figures/F5_mesh_convergence.pdf` (thesis figure, via `gen_thesis_figs.py`).
- **Tall-column physics (HEIGHT=4×w) — resolves the headline.** With a unit-height column the
  free-surface relief and the fixed-base layer overlap; a **tall column** separates them and shows the
  true picture: the in-plane residual stress is a **substratum boundary layer** (~one width deep from
  the fixed base = the implant surface) — above it the film **relieves the isotropic growth by free
  upward expansion → ~zero stress**, regardless of φ_Pg. *Within* the interface layer σ_xx tracks φ_Pg
  (corr **+1.0**). **DH ≫ CH at the interface: peak 6.4×, mean 10×.** Clinically sharp (stress
  concentrates exactly where delamination/detachment occurs). Figure `figures/F4_residual_stress_DHvsCH.pdf`
  (the headline, tall column). The "depth-resolved σ tracks φ" framing is retired in favour of "substratum-interface
  stress, DH≫CH". Tangent verified independently: `tangent_check_growth_phi.py` (Kirchhoff/Jaumann FD,
  rel-err 5e-8 at full CLSM magnitude).

## B. High-impact mechanics
- **B1+ Bilayer wrinkling (true Biot bifurcation) — ✅ ran (Abaqus, 384 elem), upgrades B1.** Stiff
  growing crust (`*USER MATERIAL`, top 2 layers, E=50000) on a soft **inert** foundation (`*ELASTIC`,
  E=2000): the crust grows in-plane but is bonded to the non-growing base → forced into compression →
  genuine surface wrinkling (`gen_film_bilayer_inp.py`, multi-wave seed, `*STATIC, STABILIZE, ALLSDTOL`).
  At γ=4 the system matrix develops **negative eigenvalues at growth-frac≈0.28** (loss of stability =
  the bifurcation); a completed γ=2 run shows the **super-linear amplitude onset** and, decisively,
  **condition-specific wavelength selection: DH selects a short wrinkle (mode 9, λ≈1.8) while CH stays
  in the long-wave global-bending mode (mode 1, λ=16)** — i.e. *only the dysbiotic crust crosses the
  wrinkling threshold*; CH merely bends. DH amplitude > CH at every growth level.
  `run_abaqus_test.sh bilayer`/`bilayer_ch` (`BILAYER_G=`) → `extract_wrinkle.py`, figure
  `F9_wrinkling_DHvsCH.pdf`. Converts the B1 free-edge "precursor" into a real growth-instability
  result; natural bridge to phase-field (Keio).
- **B0 Film strip free-edge — ✅ ran (Abaqus, 96 elem).** Depth-graded growth on a strip with a free
  right edge produces a clean **through-depth bending signature** (top S11 = −118 compression, mid =
  +43 tension, bottom = −39 — neutral axis inside the film) plus **Saint-Venant relaxation toward the
  free edge** (|S11| falls monotonically to the edge), giving out-of-plane **curl max|U3| = 0.042**.
  This is the seed for both B1 (the curl → wrinkling once γ is large) and B2 (the edge shear →
  delamination). `abaqus/run_abaqus_test.sh film` → `extract_film.py film_fs 12 8`.
- **B1 Out-of-plane deflection / buckling precursor — ✅ ran (Abaqus, 128 elem).** Free-edge film with
  a seeded bending imperfection (`gen_film_buckle_inp.py`, z-shift `amp·(z/H)·sin(πx/L)`, amp=5e-3),
  elevated growth γ=3, `*STATIC, STABILIZE=1e-4`, field output every increment. The out-of-plane
  deflection grows monotonically with growth and **DH reaches max|u₃| = 0.304 vs CH 0.247 — 23 % larger
  at identical imperfection/material**, the only difference being the φ_Pg depth profile.
  `run_abaqus_test.sh buckle`/`buckle_ch` → `extract_buckle.py` → figure `F8_buckling_DHvsCH.pdf`.
  **Honest caveat (state in thesis):** because the *free edge* relieves in-plane compression, u₃ is
  ~linear in growth — this is imperfection-amplified *bending*, not a sharp bifurcation. A true Biot
  surface-buckling onset needs either a stiffness-contrast bilayer (stiff EPS crust on a soft base) or
  compression beyond the moderate growth reached here → **Outlook**, not claimed as instability.
- **B2 Delamination — ✅ ran (Abaqus), the clinical result (⑤).** Film strip on a **cohesive base**:
  a layer of `COH3D8` elements under the film (`gen_film_delam_inp.py`), top face = film z=0 nodes,
  bottom face fixed (substratum), traction–separation law with `*DAMAGE INITIATION, CRITERION=QUADS`
  + `*DAMAGE EVOLUTION, TYPE=ENERGY` (BK mixed-mode) + `*DAMAGE STABILIZATION`. Growth-induced edge
  shear concentrates interface traction at the free edge → cohesive damage (SDEG) initiates there and
  the **delamination front propagates inward**. Strength sweep (`abaqus/run_abaqus_test.sh delam` /
  `delam_ch`, `DELAM_TN0=…`): at the operating point **t_n0 = 120**, **DH delaminates 83 % of L
  (10/12 elements) vs CH 25 % (3/12) — a 3.3× difference** with *identical* interface strength; the
  only thing that changed is the φ_Pg depth profile. ⇒ direct FEM test of dysbiosis→detachment.
  Figure `F6_delamination_DHvsCH.pdf`. Extract: `extract_delam.py film_delam 12 8`.
  | t_n0 | DH | CH | ratio |
  |---|---|---|---|
  | 60  | 92 % | 75 % | 1.2× |
  | **120** | **83 %** | **25 %** | **3.3×** |
  | 180 | 17 % | 8 % | 2.1× |
- **B3 Viscoelasticity — ✅ RAN & VERIFIED (Abaqus 2024).** Route (ii): biofilm rheology Burgers creep
  fit → **exact 2-term Prony** via `coupling_prototype/prony_from_burger.py`
  (GM=5,etaM=3000,GK=10,etaK=100 → g1=0.321@6.64s, g2=0.629@903s; deviatoric-only, floor g_inf=0.05).
  One-element simple-shear relaxation deck `abaqus/biofilm_viscoelastic.inp` (`*VISCO`, hold shear)
  COMPLETED; S13 relaxes from the glassy value to the floor: **final/instantaneous = 0.0544 vs theory
  0.0543**, and **RMS(FE − analytic Prony) = 9e-4** over the whole curve (relaxation clock at the
  strain-application instant). Figure `figures/F7_viscoelastic_relaxation.pdf`. The
  long-time relaxation is exactly the fluid-like complement of Böl 2014 (`notes/bol2014_validation_paragraph.md`).
  *Outlook route (i): carry the viscous strain in STATEV inside the φ_Pg growth UMAT for coupled
  growth+relaxation; pairs with ④ G(t) from rheology.*

## C. Data / ideas
- **C4 Patient-specific detachment risk — ✅ ran (Abaqus sweep), the clinical translation.** The
  cohesive-delamination model run across the growth driver φ_Pg gives a monotone **detachment-risk
  curve** on the clinically relevant branch (φ≤0.14: delam 0→8→33 %). Mapping the **10 Dieckow patients**
  by their Bacteroidia(Pg) load (0.001–0.122, median 0.054) onto it stratifies the cohort: **nine
  patients sit below the delamination onset (~φ 0.12, low risk); a single high-Pg patient (φ=0.122)
  reaches ~26 % predicted detachment** — a mechanics-derived, per-patient risk score from routine 16S
  Pg load. `abaqus/run_risk_sweep.sh` (PHIS, RISK_GAMMA) → `risk_sweep.txt`; figure `F10_patient_risk.pdf`.
  Honest caveat: the high-φ branch of the sweep is non-monotone (cohesive damage-snap, stabilization-
  sensitive) so only the clean low-φ clinical branch is used; absolute % is model-parametric, the
  patient *ordering* is the robust output.
- **C1 Biofilm mechanics literature — [me, ④ scaffold].** Oral/biofilm shear modulus **G ~ 10–10³ Pa**
  (species- and shear-history-dependent), strongly **viscoelastic** (creep/relaxation), often Maxwell
  / Burgers / power-law. Detachment driven by interfacial shear ~ 0.1–10 Pa wall shear in flow.
  *(Verify cites: Stoodley/Klapper/Wilking-Brenner; numbers are order-of-magnitude established.)*
  → justify the compressible-neo-Hookean-for-Fe choice as the fast/elastic limit; use G~100 Pa for
  any quantitative claim; report σ/μ where μ is uncertain.
- **C2 CLSM growth-rate time series — [run-done].** From `zprofiles_all_ti.csv` (days 1,6,10,15,21):
  **DH thickening rate μ≈0.046/day vs CH μ≈0.003/day (≈15× faster)** — an independent quantitative
  signature of dysbiosis, consistent with DH>CH residual stress. Feeds the growth-rate calibration of γ.
- **C3 CS/DS depth data — [checked: NOT available].** Both `zprofiles_all_ti.csv` and `fish_3d/`
  contain **only CH and DH**. CS/DS would need new CLSM imaging. → state the CH-vs-DH comparison as
  the available contrast; CS/DS as an imaging-dependent extension.

## Run order (your terminal)
```bash
cd masterarbeit_ansys_fem/coupling_prototype
abaqus/run_abaqus_test.sh patch     # A1 patch test (uniform stress)
abaqus/run_convergence.sh           # A2/A3 mesh convergence (energy norm rigorous)
abaqus/run_abaqus_test.sh film      # B precursor: free-edge stress concentration + U3
# then build B1/B2/B3 from the recipes above (I'll iterate with you on results)
```
