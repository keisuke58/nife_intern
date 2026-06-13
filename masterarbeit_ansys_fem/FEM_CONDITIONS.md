# FEM conditions — single source of truth (defence reference)

Complete specification of **every** finite-element model in the peri-implant / peri-implantitis FEM
track: materials, boundary conditions, loads, interfaces, element types, solver. Written so that any
examiner question ("what stiffness did you use for the PDL?", "how is the bite applied?", "what holds
the bone?") is answered here with the exact value and its source in the code.

Solver: **Abaqus/Standard 2024** (local, `~/DassaultSystemes/SIMULIA/Commands/abaqus`, `cpus=1`, shared
DSLS license 172.17.36.96). Meshing: **gmsh 4.15.1** (conda env `gmsh_env`). Units throughout: **mm,
N, MPa** (so stresses are in MPa, forces in N, lengths in mm). Time in the growth/load steps is
pseudo-time (quasi-static).

---

## 1. Material properties (linear elastic isotropic unless noted)

The **single canonical material set** (defined in `coupling_prototype/tier2b_real/build_assembly.py`,
`periimplantitis_coupon.py`, `gen_transmucosal_axi.py`). Every model below draws from this table.

| Material | E (MPa) | ν | Role | Basis |
|---|---:|---:|---|---|
| **TI** (Ti-6Al-4V) | 110 000 | 0.34 | implant fixture + abutment | titanium alloy, standard dental-implant FEM (Geng 2001; Sevimay 2005) |
| **CORTICAL** bone | 13 700 | 0.30 | dense outer / lamina-dura shell (1.8 mm) | cortical mandibular bone (Lin 2010; Sevimay 2005) |
| **CANCELLOUS** bone | 1 000 | 0.30 | trabecular core | low-density trabecular bone (type III–IV) |
| **BONE** (single-layer) | 13 700 | 0.30 | used only in the preserved `tier2b_real` job | cortical-equivalent (thin alveolar crest is cortical-dominated) |
| **DENTIN** | 18 000 | 0.31 | natural neighbour tooth (tooth-24) | human coronal dentin (Lin 2010) |
| **PDL** | 50 | 0.45 | 0.25 mm periodontal ligament, neighbour tooth | linear-elastic PDL idealisation (Cattaneo 2005 use 50–69 MPa; real PDL is nonlinear → **idealisation, noted**) |
| **CROWN** (ceramic) | 95 000 | 0.30 | load-bearing restoration | lithium-disilicate-class glass-ceramic (e.max ≈ 95 GPa). Zirconia variant = 210 GPa (design sweep) |
| **GINGIVA** (mucosa) | 3 | 0.45 | peri-implant soft-tissue cuff (transmucosal model only) | oral mucosa, ~MPa-scale, near-incompressible |
| **BIOFILM** | 1.0 | 0.45 | dysbiotic growth collar (+ growth eigenstrain) | soft hydrogel-like; modulus is **uncertain (Pa–MPa, 6 orders)** → results given as *ratios*, not absolutes |

**Notes that matter for the defence**
- **PDL and biofilm are deliberate linear-elastic idealisations.** Real PDL is hyperelastic/viscoelastic
  and bilinear; we use a single secant modulus (50 MPa) because the study quantity is the *contrast*
  implant-vs-tooth and crown moment arm, not the absolute PDL strain. The biofilm modulus is unknown to
  ~6 orders of magnitude, so **every biofilm/dysbiosis result is reported as a dimensionless ratio**
  (DH/CH ≈ 2.3–3.3× across all models), never as an absolute stress.
- **Crown stiffness barely matters for peri-implant bone stress** (it is a near-rigid load path); what
  matters is the load *height* (moment arm). The crown material only changes the crown's own stress.

### Growth eigenstrain (biofilm swelling) — the one non-mechanical "load"
Biofilm growth is imposed as an **isotropic volumetric eigenstrain** via thermal analogy:
`*EXPANSION α = EPS_GROWTH = 0.19` with a unit temperature rise (`*TEMPERATURE 0 → 1`) in step 1.
ε_growth = 0.19 (≈ DH-calibrated swelling from CLSM; CH uses the same field driven by measured Pg depth).
This is the *only* driver in the growth step; no external force there.

---

## 2. Model families (each row = one solved configuration)

| Model / file | Element | Dim | Geometry | Purpose |
|---|---|---|---|---|
| `build_assembly.py → tier2b_real` | C3D4 | 3D | real mandible crop + real tooth-23 root-analog Ti + tooth-24 + PDL | preserved baseline (near-axial load) |
| `… → tier2b_generic` | C3D4 | 3D | **generic Ø4.1×10 screw** in real mandible + tooth-24 | primary coupled model (ISO-14801 oblique) |
| `… → tier2b_crown` | C3D4 | 3D | generic screw + **parametric ceramic crown** (load at occlusal table) | crown moment-arm study |
| `… → tier2b_crownreal` | C3D4 | 3D | generic screw + **real-molar voxel crown** | validation of the parametric crown (−10% on bone peak) |
| `periimplantitis_coupon.py` | C3D10 | 3D | ISO-14801 screw-in-holder, variable bone level | progression, design & C/I-ratio sweeps |
| `gen_transmucosal_axi.py` | CAX4 | axisym | Ti + bone + mucosa + sulcus + biofilm | biofilm location (between Ti & gingiva) |
| `gen_implant_axi_inp.py` | CAX4 | axisym | cylinder implant | hoop (3D) effect ∝ 1/R₀ |
| `gen_implant_inp.py / _phi / _rve` | CPE4 | plane-strain | thread vs flat | growth-stress at the thread interface |
| `gen_implant_3d_inp.py` | C3D8 | 3D | true helical screw | helical stress band, occlusal fatigue |
| `gen_implant_poro_inp.py` | CPE4P | plane-strain | thread + pore fluid | drained vs undrained interface |
| `implant_coupon.py` | C3D4/C3D10 | 3D | ISO-14801 design coupon | diameter/length/pitch sweep |

The **growth-stress plane-strain/axisymmetric models use normalised stress units** (biofilm modulus
unknown) and report dysbiosis *ratios*. The **tier2b assembly and the ISO-14801 coupons use physical
MPa** with the material table above.

---

## 3. Geometry & dimensions

**Generic implant** (`mesh_generic_implant.py`): Ø4.1 mm × 10 mm body, 1.0 mm thread pitch, core radius
1.55 mm / crest radius 2.05 mm, transmucosal abutment radius 1.8 mm to z = 32.5 mm (platform/crest at
z = 29 mm). Mesh edge 0.45 mm. Standard Straumann/Nobel-class screw idealised as concentric thread rings
(body-of-revolution).

**Crown** (`mesh_crown.py`): hollow ceramic cap sheathing the abutment; cervical margin z = 31 mm
(crest + 2 mm = gingival level), occlusal table z = 38 mm (matched to the tooth-24 occlusal plane → **not
supra-occluded**), clinical crown height ≈ 7 mm, internal bore 1.85 mm (0.05 mm cement gap). Real-molar
variant (`mesh_crown_voxel.py`): OpenJaw P1_Tooth_30 clinical crown, voxelised at 0.35 mm.

**ISO-14801 coupon** (`periimplantitis_coupon.py`): Ø4.1 × 10 mm screw, 0.8 mm pitch, 0.40 mm thread
depth, abutment 8 mm, bone holder radius 4.5 mm. `expose` = marginal bone loss below the platform (mm).

**Mandible crop** (real coupled model): X[−76,−59], Y[−47,−37.5], Z[15,31] mm (real Open-Full-Jaw
Patient-1 with real alveolar sockets as voids). Cortical shell / lamina dura = 1.8 mm. PDL = 0.25 mm.

---

## 4. Boundary conditions

| Model | Fixed (Dirichlet) | DOF |
|---|---|---|
| tier2b assembly | bone nodes on the **artificial crop faces** (z ≤ Z0, x ≤ X0, x ≥ X1, y ≤ Y0, y ≥ Y1) | U1=U2=U3=0 (`*BOUNDARY FIXED, 1, 3`) |
| ISO-14801 coupon | bone holder **base (z ≤ −0.6)** and **outer wall (r ≥ R_holder−0.4)** | U1=U2=U3=0 (`CLAMP, 1, 3`) |
| axisymmetric (CAX4) | bone base; axis symmetry implicit in CAX | radial on axis |
| plane-strain (CPE4) | base adhered; sides free (採用) / periodic RVE (`*EQUATION`) in the closure model | — |

The crop-face fixity is a **St-Venant truncation**: the crop is taken large enough that the crestal
region of interest is unaffected (verified by the periodic-RVE closure giving the same thread-stress
conclusion → not a free-edge artefact).

---

## 5. Loads (occlusal bite — ISO 14801)

- **Magnitude:** 100 N resultant per crown (physiological molar bite; ISO 14801 worst-case is 30°).
- **Direction:** **30° oblique** for the generic/crown/coupon jobs → lateral/axial = tan 30° = **0.577**
  (`*CLOAD` components: F3 = −100·cos30 = −86.6 N axial, F1 = +100·sin30 = +50 N lateral). The preserved
  `tier2b_real` job keeps a near-axial 0.2 ratio.
- **Application point (this is the crown moment-arm story):**
  - bare abutment / `tier2b_generic`: load on the **abutment-top nodes** (z = 32.5).
  - `tier2b_crown` / `crownreal`: load on the **crown occlusal table** (z ≈ 38) → adds the crown-height
    moment arm (~3.4× bending at the neck).
  - ISO-14801 coupon with `crown_h > 0`: load applied at a **reference node** at the occlusal height,
    rigidly tied to the abutment-top nodes by `*COUPLING / *KINEMATIC` (textbook ISO-14801 offset load).
- **Two-step procedure:** Step 1 = biofilm growth (eigenstrain, no external force). Step 2 = occlusal
  load (`*STATIC`, NLGEOM=NO for bonded; NLGEOM=YES + `*STATIC, STABILIZE` for frictional contact).

---

## 6. Interfaces

| Interface | Type | Setting |
|---|---|---|
| implant ↔ bone (osseointegrated) | `*TIE` ADJUST=NO | POSITION TOLERANCE 2.8 (generic, screw < socket) / 1.0 (root-analog) |
| PDL-outer ↔ bone socket (tooth) | `*TIE` ADJUST=NO | POSITION TOLERANCE 1.0; PDL inner shares dentin nodes (conforming) |
| crown seat ↔ abutment top | `*TIE` ADJUST=NO | POSITION TOLERANCE 0.5 (bore-roof disk r < 2 mm ↔ abutment top) |
| bone-implant (peri-implantitis contact variant) | `*CONTACT`, general, ALL EXTERIOR | `*FRICTION μ = 0.3`, HARD pressure-overclosure |

`ADJUST=NO` is deliberate: moving slave nodes onto the master would collapse the 0.25 mm PDL / thin
implant-tip tets; the small initial socket gap is absorbed into the rigid tie instead. The generic
screw (Ø4.1) is narrower than the natural socket (buccolingual ≈ 8.5 mm) → bonded to the socket walls
= **explicit idealisation of immediate, fully-integrated placement** (stated as a limitation).

---

## 7. Mesh & solver settings

- **Element orders:** C3D4 (linear tet) for the large coupled assembly (~270 k elems); **C3D10 (quadratic
  tet)** for the ISO-14801 coupons (linear tets under-predict thread-root concentration by ~9–15%, so the
  coupons use C3D10). gmsh tet10 → Abaqus C3D10 needs the **last two mid-nodes swapped** and
  `Mesh.SecondOrderLinear=1` (straight-edge mid-nodes → no negative Jacobian).
- **Sliver removal:** tets with volume < 1e-4 mm³ are dropped; negative-volume tets are re-wound.
- **Solver:** Abaqus/Standard, direct solver, `cpus=1`, geometric linearity (NLGEOM=NO) for bonded
  static; NLGEOM=YES with `*STATIC, STABILIZE=1e-4` for the frictional micromotion variant.
- **Convergence:** mesh-refinement check on the implant coupon = ±7% on the thread-root peak between
  refinement levels (reported as study (B) of the rigour panel).

---

## 8. Key quantitative results (with their conditions)

| Result | Condition | Value |
|---|---|---|
| Dysbiosis stress ratio (DH/CH) | every growth model | **2.3 – 3.3×** (robust across plane-strain/axisym/3D/RVE) |
| Crown moment arm → crestal bone peak | tier2b_crown, 100 N/30°, load z 32.5 → 38 | 55 → 88 MPa (**×1.6**); neck Ti 146 → 180 MPa |
| Real-molar crown validation | tier2b_crownreal vs tier2b_crown | 79 vs 88 MPa (**−10%** → parametric crown adequate) |
| C/I-ratio feedback | coupon sweep + resorption ODE, same severity | crowned reaches point-of-no-return **34 mo vs 55 mo** (38% sooner) |
| Marginal-bone-loss → stress | coupon, bone 2 → 8 mm | crest 14 → 35 MPa; stiffness 2.7 → 0.7 N/µm |
| Micromotion (de-integrated) | general contact, 100 N | ~7–8 µm ≪ Brunski 150 µm threshold |

---

## 9. Idealisations & limitations (state these proactively)

1. **Linear elastic** materials (no plasticity/yield): Ti peak 180–230 MPa stays well below Ti-6Al-4V
   yield (~800 MPa), so elasticity is valid for the implant; bone/biofilm are screening-level.
2. **PDL & biofilm moduli** are single linear values (real = nonlinear / uncertain) → results are
   **ratios and orderings**, not absolute strains.
3. **Osseointegration = perfect bond** (`*TIE`); partial BIC handled separately by the frictional-contact
   variant.
4. **No soft tissue in the load path** (mucosa omitted except in the transmucosal model): soft tissue is
   ~10⁴× softer than bone, carries negligible occlusal load → does not change bone stress; its biological
   role (inflammation) is carried by the disease ODE, not the mesh.
5. **Voxel crown** (real-molar) has a blocky surface — used for the bone-stress validation and the
   stressed Panel-B body; the smooth real-tooth STL is the Panel-A illustration.
6. **Generic screw bonded to a wider natural socket** = immediate, fully-integrated idealisation.
7. **Quasi-static, single load cycle** (fatigue/remodelling handled by the separate ODE layer).

---

## 10. Reproduce

```bash
# build (gmsh_env) then solve (Abaqus) from FEM/tier2b_real/
conda activate gmsh_env; export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
python mesh_generic_implant.py                 # generic screw
python mesh_crown.py                           # parametric ceramic crown
python mesh_crown_voxel.py 0.35                # real-molar voxel crown
python build_assembly.py cache_implant_generic.npz tier2b_crown      # (or tier2b_crownreal)
abaqus job=tier2b_crown input=tier2b_crown.inp interactive cpus=1
abaqus python extract_tier2b.py tier2b_crown   # -> *_field.json (per-element vM, both steps)
# ISO-14801 coupon sweeps:
bash run_pimp.sh        # progressive bone loss
bash run_crown_ci.sh    # crown-to-implant ratio x bone loss
```

Material values live in the `MATS` dict of each generator; loads/BCs in the `*STEP` block of the
written `.inp`. This document is the authoritative copy — if a value changes in code, update it here.
