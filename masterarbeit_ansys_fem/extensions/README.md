# Extension ideas — all ten realised (2026-06-13)

Follow-on research seeded from the verified FEM coupling. Each produced a real running result + figure.
**Tier A** = thesis-grade (Abaqus / verified numerics). **B/C** = research prototypes (Keio/paper track).
**D** = career transfer note. Figures land in `../figures/F{9..17}_*.pdf`.

| # | Idea | Result | Status | Fig |
|---|------|--------|--------|-----|
| **A1** | Bilayer Biot wrinkling (upgrades B1 precursor) | Negative system eigenvalues at growth-frac 0.28 (bifurcation); **DH wrinkles (mode 9, λ≈1.8), CH only bends (mode 1)** — only dysbiosis crosses the threshold | ✅ Abaqus (384 elem) | F9 |
| **A2** | Poroelastic drainage (Böl pressure-restricted limit) | 1-D Terzaghi: FD vs analytic **RMS 1.8e-3**; residual stress half-relaxes at t=0.197τ, ~95% by τ. Elastic (t→0, thesis) ↔ drained (t→∞, Böl) | ✅ Python verified; `abaqus/gen_poro_inp.py` = `*SOILS` C3D8P scaffold (needs poro material-card tuning) | F15 |
| **A3** | Patient-specific detachment risk | 10 Dieckow patients on the mechanics risk curve: **9 low-risk + 1 high-Pg patient (φ=0.122) at ~26%** | ✅ Abaqus sweep (`abaqus/run_risk_sweep.sh`) | F10 |
| **B1** | Phase-field interface fracture | Mesh-independent crack (Nx 80↔240 diff **1.6e-3**); **DH cracks (len 0.20), CH stays sub-threshold** — sharp dysbiosis/health separation | ✅ Python (AT2); production = Abaqus UEL | F16 |
| **B2** | Multiscale WLC constitutive law | 8-chain WLC network vs neo-Hookean: matches at small strain, **1.3× strain-stiffening lock-up** the crude model misses | ✅ Python; the law a Keio FE² scheme calls | F12 |
| **B3** | ML surrogate of the inverse problem | Forward surrogate **R²=0.97, RMSE 1%**; inversion recovers Pg-at-substratum from observed detachment | ✅ Python (sklearn); the map a QA/ML optimiser drives | F13 |
| **C1** | Inverse design / probiotic steering | Same total Pg load, steered from substratum to surface → **92% lower detachment driver** | ✅ Python (LP) | F11 |
| **C2** | Two-way PDE×FEM (mechanotransduction) | Stress-gated carrying capacity: substratum Pg **1.0→0.17**, Pg CoM up 0.12, peak interface stress **−32%** — mechanics self-limits sequestration | ✅ Python | F14 |
| **C3** | Residual stress from REAL CLSM FISH data | Measured-profile detachment driver **J(day) diverges**: CH falls, DH rises, cross ~day6, endpoint **DH/CH 1.41**. Total Pg does NOT separate (CH 0.164 vs DH 0.150) — **depth POSITION does** | ✅ Python on real data (species-resolved Böl pipeline) | F17 |
| **D** | Transfer to semiconductor processing | One-to-one map: growth eigenstrain→stress→delamination ≅ thin-film stress / dicing-street chipping / die-attach delamination (DISCO) | ✅ note (`D_disco_transfer.md`) | — |

## Run
```bash
# Python prototypes (TeX Live on PATH for usetex figures)
PATH=/home/nishioka/texlive/2025/bin/x86_64-linux:$PATH python extensions/<name>.py
# Abaqus (A1, A3): from coupling_prototype/abaqus
./run_abaqus_test.sh bilayer ; ./run_abaqus_test.sh bilayer_ch     # A1
./run_risk_sweep.sh                                                # A3
```

## Where each lands
- **Thesis body / B1-paragraph upgrade**: A1 (real wrinkling bifurcation) directly replaces the B1
  "precursor" caveat. A2 strengthens the §5.4 viscoelastic/limits paragraph (drainage timescale).
- **Thesis Outlook**: A3, C1, C3 (clinical), B2 (constitutive).
- **Keio / Muramatsu continuation**: B1 phase-field, B2 FE², B3 QA-ML, C2 morphogenesis.
- **Paper**: C1 (designed intervention), C3 (real-data temporal divergence).
- **DISCO interview**: D.
