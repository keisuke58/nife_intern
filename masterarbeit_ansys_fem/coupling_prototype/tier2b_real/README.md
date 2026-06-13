# Tier-2(b): full real-shape coupled implant + tooth + alveolar-bone FEM

A coupled Abaqus model built entirely from **real Open-Full-Jaw Patient-1 anatomy**, the realistic
counterpart to the parametric Tier-2(a) bone block (`../abaqus/gen_tier2_bone_inp.py`).

## What it is
A single multi-material assembly, occlusally loaded, with the load transmitted through the **shared
alveolar bone** so the implant and the adjacent tooth are genuinely mechanically coupled:

| part | source | material | coupling |
|------|--------|----------|----------|
| BONE | real `P1_Mandible.stl` cropped around teeth 23/24 (real alveolar sockets are voids) | 13.7 GPa | — |
| DENTIN | real `P1_Tooth_24.stl` solid | 18 GPa | tooth root |
| PDL | 0.25 mm conforming offset layer on the tooth-24 surface (nodes shared with dentin) | 50 MPa | tooth↔bone via PDL |
| TI | real `P1_Tooth_23.stl` solid, titanium — natural tooth "extracted", root-analog implant | 110 GPa | **osseointegrated, tied directly to bone (no PDL)** |
| BIOFILM | crestal dysbiotic-growth collars carved from the bone crest | 1 MPa, ε=0.19 | — |

Couplings: `*TIE` (ADJUST=NO) PDL-outer↔bone-socket(24) and implant↔bone-socket(23).
Steps: (1) dysbiotic biofilm growth eigenstrain; (2) occlusal load (60 N/crown, oblique).

## Pipeline (run in the `gmsh_env` conda env with `LD_LIBRARY_PATH=$CONDA_PREFIX/lib`)
1. `prep_meshes.py`   — volume-mesh + cache BONE (real mandible crop), DENTIN, IMPLANT (gmsh 4.15).
2. `build_assembly.py`— PDL offset layer, global assembly, sliver cleanup, TIE surfaces → `tier2b_real.inp`.
3. solve: `abaqus job=tier2b_real cpus=1 interactive`  (≈2.5×10⁵ C3D4, solves in ~1 min).
4. `extract_tier2b.py` (`abaqus python`) → `tier2b_real_field.json` (per-element vM, both steps).
5. `analyze_coupling.py` — peri-implant vs peri-tooth bone stress; figure `fig_tier2b_real.py`.

Working files live under `/home/nishioka/IKM_Hiwi/FEM/tier2b_real/` (not a git repo); these copies
are the version-controlled record.

## Generic standard screw-form implant variant (`tier2b_generic`)
`mesh_generic_implant.py` builds a **generic parametric titanium screw** (Ø4.1 mm × 10 mm, 1.0 mm
pitch V-thread + transmucosal abutment; body-of-revolution, concentric thread-ring idealisation) in
the real mandible frame at the tooth-23 axis, replacing the patient root-analog. Re-run the same
pipeline with the new cache + job name (originals untouched):

```
python mesh_generic_implant.py                                  # -> cache_implant_generic.npz
python build_assembly.py cache_implant_generic.npz tier2b_generic
abaqus job=tier2b_generic cpus=1 interactive
abaqus python extract_tier2b.py tier2b_generic
python analyze_coupling.py tier2b_generic
python ../../extensions/fig_tier2b_real.py tier2b_generic       # -> figures/fem_tier2b_generic.pdf
```

`build_assembly.py` enlarges the tooth-23 master radius (4.8 mm) and the implant `*TIE` position
tolerance (2.8 mm) only for the generic job, because the standard Ø4.1 screw is narrower than the
natural (buccolingually ~8.5 mm) socket — the screw is bonded to the socket walls, an **explicit
idealisation** (a real placement would be a healed/drilled ridge).

**Generic-screw result**: peak stress is now carried by the **titanium thread roots** (occlusal vM
≈79 MPa vs 49 MPa for the smooth root-analog — the classic implant thread stress-concentration). The
peri-implant vs peri-tooth crestal-bone contrast stays modest (implant 5.9 ≈ tooth 5.9 MPa at the
crest; ratio 0.85 over the interface shell), reflecting the partial socket engagement of a standard
screw in this site. Figure `figures/fem_tier2b_generic.pdf`.

## Result (honest, root-analog `tier2b_real`)
The fully real-geometry coupled model **solves successfully** — the methodological goal. With a
literature-standard linear PDL (50 MPa) the peri-implant vs peri-tooth crestal-bone contrast is
**modest**: at the crest (z∈[26,28.5]) the osseointegrated implant concentrates slightly more stress
(≈6.6 vs 6.1 MPa) while the PDL-supported tooth distributes load deeper along the root (z∈[16,20]:
2.1 vs 1.6 MPa) — qualitatively the peri-implant marginal-bone-loss signature, but far weaker than the
idealized parametric Tier-2(a). PDL modulus was **not** softened to inflate the contrast.
Suitable as a thesis-Outlook demonstration, not a quantitative claim.
