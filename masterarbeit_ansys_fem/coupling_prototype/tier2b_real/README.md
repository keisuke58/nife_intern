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

### Level-up (the adopted generic model)
The `tier2b_generic` job additionally carries two realism upgrades over the root-analog:
1. **Two-layer bone** — cortical shell / lamina dura (within 1.8 mm of the real outer & socket-wall
   surfaces, 13.7 GPa) vs cancellous core (1.0 GPa). This thin alveolar crop is cortical-dominated
   (cortical 232792 vs cancellous 6406 tets) — anatomically reasonable; the cancellous core is deep
   and lightly loaded (≈0.5 MPa), so it barely shifts the interface result, but the distinction is
   carried explicitly.
2. **ISO 14801-style 30° oblique occlusal load, ~100 N/crown** (lateral/axial = tan30 = 0.577),
   replacing the earlier near-axial load — the standardised, clinically dominant condition.

**Leveled-up generic result**: under 30° oblique loading the **crestal peri-implant bone reaches
≈20 MPa** (vs ≈6 MPa near-axial — oblique load dominates marginal-bone stress), with the implant
slightly exceeding the tooth at the crest (z∈[26,28.5]: implant 20.3 vs tooth 19.1 MPa; interface
shell ratio 1.03) — the peri-implant marginal-bone-loss signature now emerges. Peak stress sits in
the **titanium thread roots** (≈146 MPa, well within Ti yield) — the classic implant thread stress
concentration. The standard Ø4.1 screw is narrower than the natural (buccolingually ~8.5 mm) socket,
so it is bonded to the socket walls (immediate-placement-like partial engagement — an explicit
idealisation). Figure `figures/fem_tier2b_generic.pdf`.

## Implant DESIGN study — ISO 14801-style coupon (`implant_coupon.py`)
The thread-dimension sweep, C3D4-vs-C3D10 accuracy and axial-vs-oblique comparison are implant-LOCAL
properties, best isolated on a standard coupon rather than re-running the full mandible. A parametric
Ti screw is osseointegrated (conforming via OCC `fragment`) in a bone holder cylinder, embedded 3 mm
below the platform, clamped, and loaded ~100 N at a chosen angle (ISO 14801).

```
bash run_coupons.sh        # builds 10 coupons (gmsh_env), solves + extracts (Abaqus) -> coupon_results.jsonl
python ../../extensions/fig_implant_design.py   # -> figures/fem_implant_design.pdf (4 panels)
```
- `implant_coupon.py D L pitch taper order angle tag` — body-of-revolution screw + bone cylinder,
  fragmented to a conforming bonded mesh; `order` 1=C3D4 / 2=C3D10. **C3D10 note**: gmsh tet10 node
  order differs from Abaqus C3D10 in the last two mid-edge nodes (swap cols 8,9), and
  `Mesh.SecondOrderLinear=1` is required (straight-edge quadratic) to avoid curved-edge negative
  Jacobians; sliver tets (<1e-4 mm³) are dropped.
- `extract_coupon.py tag L` (abaqus python) → thread-root p99 vM, max-Ti vM, bone/crestal p95, tip
  displacement (→ stiffness).

**Results** (Ø4.1×10 mm, 0.8 pitch, cylindrical baseline; thread-root p99 vM):
| study | finding |
|---|---|
| **diameter (dominant)** | Ø3.5→4.1→4.8: thread **171→104→60 MPa**, stiffness 1.2→2.2→4.1 N/µm — wider implant strongly lowers stress & raises stiffness |
| length 8→12 mm | thread 101→98 MPa — negligible (diminishing returns past ~8 mm) |
| pitch 0.6→1.0 mm | 100→101 MPa — secondary |
| taper 0→0.3 | 104→119 MPa — narrower apex concentrates more |
| **load angle (ISO 14801)** | axial→30° oblique: thread **×9** (12→104 MPa), crestal bone ×7, displacement ×27 — oblique loading dominates |
| **mesh order** | C3D4→C3D10: thread-root **+9 %**, max-Ti **+15 %** — linear tets under-predict the concentration; C3D10 used |

Figure `figures/fem_implant_design.pdf` (4 panels). Peak Ti stresses (60–230 MPa) stay well within
Ti-6Al-4V yield (~800 MPa). The coupon isolates implant design; the full-mandible `tier2b_generic`
provides the anatomical-coupling context.

## Peri-implantitis DISEASE-process studies (4 pillars)
Beyond implant *design*, these model the disease itself --- progressive marginal bone loss --- on the
same coupon. Scripts: `periimplantitis_coupon.py` (parametric bone level, bonded), `contact_coupon.py`
(debonded, frictional general contact), `extract_pimp.py` / `extract_pimp_field.py` /
`extract_contact.py`, runner `run_pimp.sh`. Figures: `fig_periimplantitis*.py`.

**(1) Progressive marginal bone loss → stress feedback** (`run_pimp.sh`, bone level 2→8 mm):
crestal-bone $\sigma_\mathrm{vM}$ rises $14\to35$ MPa and coupon stiffness collapses $2.7\to0.7$ N/µm
(×3.7) — an **accelerating vicious cycle / point-of-no-return** (sharp jump at 8 mm). Fig
`fem_periimplantitis_progression.pdf` (A).

**(3) Biofilm severity sets the rate** (same fig, B): the dysbiotic interface drives resorption ≈2.5×
faster than the commensal (the Chapter-5 contrast applied as the time-axis rate), so DH reaches the
accelerating regime far sooner. (Severity enters the rate, not the mesh.)

**(2) Mechanobiological remodelling (Frost/Carter mechanostat)** (`fig_periimplantitis_remodel.py`):
strain-energy density SED$=\sigma_\mathrm{vM}^2/2E$ concentrates at the **crestal** bone on the loaded
(buccal) side → the FEM predicts resorption initiates there and progresses as a saucer-shaped defect
("saucerisation"). Fig `fem_periimplantitis_remodel.pdf`.

**(4) Loss of osseointegration → micromotion (Brunski)** (`contact_coupon.py`, 0% BIC, µ=0.3):
a fully debonded implant micro-moves only ≈7–8 µm at 100 N across 4–8 mm bone loss — **well below the
Brunski ~150 µm** fibrous-encapsulation threshold; sub-threshold even at ~6× bite force. Honest
conclusion: early/moderate peri-implant bone loss is **biofilm-driven, not micromotion-driven** ---
mechanical instability appears only at near-total loss / parafunction. Fig
`fem_periimplantitis_micromotion.pdf`. (general contact, `STABILIZE`; micromotion = implant-vs-socket
nodal $\Delta U$; contact-output CSLIP/CPRESS dropped — invalid for this option.)

## Time-series coupling: ecology(t) -> predicted bone-loss trajectory (no new solves)
Closes the loop with existing longitudinal data, reusing the study-(1) stress-feedback response
A(loss)=crest(loss)/crest(2mm). Resorption ODE: d(loss)/dt = k·severity·A(loss).
- `fig_periimplantitis_invivo.py` -> `fem_periimplantitis_invivo.pdf` **(preferred, in-vivo)**: the
  clinically-validated guild dysbiosis index GDI=log(Fuso+Bact+Clos)−log(Bac+Act+Neg) computed
  per-patient from REAL Dieckow 2024 abundances (10 pt × 3 wk) drives a **per-patient bone-loss
  trajectory + risk ranking** (extends the thesis per-patient J ranking). Honest caveat: Dieckow is a
  longitudinal *development* cohort (all GDI<0, sub-clinical) -> the result is RELATIVE per-patient
  stratification; the absolute disease range comes from the Joshi-validated DI–severity link (ρ=0.46).
- `fig_periimplantitis_botelho.py` -> `fem_periimplantitis_botelho.pdf` **(strongest in-vivo)**: same
  GDI driver on Botelho 2021 (PRJNA725874, `data/prjna725874/phi_guild.npy`, 15 pt × 7 tp / 12 wk) --
  a genuine **periodontitis** longitudinal cohort (dysbiotic bone-loss disease, peri-implantitis
  analogue), so more disease signal + a longer window than Dieckow. Per-patient GDI(t) → bone-loss
  trajectory + ranking (P8/P2/P13 highest-risk).
- `fig_periimplantitis_timeseries.py` -> `fem_periimplantitis_timeseries.pdf` (in-vitro, secondary):
  the gLV/Hamilton-fitted 5-species attractor trajectories φ_Pg(t) (CS/CH/DS/DH). **Driver caveat**:
  raw φ_Pg mis-ranks severity here — Pg is a minor member and the dysbiotic states differ partly by Pg
  STRAIN virulence (W83 vs ATCC), not abundance; so GDI (not φ_Pg) is the sound severity metric.

**Data-choice summary**: gLV-fit (in-vitro) = best dynamics + time + dysbiosis states but φ_Pg mis-ranks
(strain virulence); in-vivo Dieckow = real patients + longitudinal + validated GDI but healthy cohort
(limited disease range); Joshi = real disease range + validated DI but cross-sectional (no time).
Best driver = GDI; ideal = Dieckow/Botelho dynamics × Joshi absolute severity.

## Host INFLAMMATORY response layer (the biologically complete cascade)
`fig_periimplantitis_inflammation.py` -> `fem_periimplantitis_inflammation.pdf`. Inserts the immune
stage the direct GDI->loss model omitted: coupled ODEs
   dysbiosis B -> inflammation I (cytokines) -> osteoclast C (RANKL/OPG sigmoidal switch) -> bone loss L,
with mechanical-overload synergy A(loss) from study (1), driven per-patient by the real Botelho GDI.
The cytokine **threshold** turns the disease into a **tipping-point / bistable** process (clinically
episodic, not gradual): sub-threshold patients stay stable (~0.3 mm), supra-threshold patients run away
(3–4 mm at 36 mo). Panel A shows the time-lagged cascade (inflammation in weeks, osteoclast lag, bone
loss in months); panel B the threshold-gated outcome. Pure ODE, no FEM solves. NB: it is a host-response
*bone-loss* model — inflammation enters as the driver stage; rate constants are physiologically ordered
but illustrative (relative progressor/stable split is the robust claim).

## Accuracy upgrade: calibrated priors + Monte-Carlo UQ + sensitivity
`fig_periimplantitis_uq.py` -> `fem_periimplantitis_uq.pdf`. Turns the deterministic illustrative
cascade into a probabilistic one (on-brand with the thesis's Bayesian core):
- **literature-anchored priors** (lognormal): cytokine resolution gI~0.5-2/wk, osteoclast turnover
  gC~1/(2-4 wk), Hill n 4-10, mechanical synergy lam 0.5-3, resorption kL calibrated to clinical
  peri-implantitis progression;
- **Monte-Carlo (N=600)** over priors + per-patient GDI timepoint scatter -> per-patient 36-mo bone-loss
  **median + 90% credible interval** (panel A) and **P(crossing the point-of-no-return ≤36 mo)** as a
  probability (panel B);
- **global sensitivity** (panel C): osteoclast activation/turnover (kC, gC) and resorption gain (kL)
  dominate the outcome variance — the quantities to measure next; Hill n and mechanical-synergy lam are
  nearly irrelevant. (For the moderate Botelho cohort P(progress) stays ≤0.12 — honest: none is likely
  to reach the mechanical point-of-no-return within 36 mo.)

## Anatomically faithful transmucosal model (where the biofilm sits)
`gen_transmucosal_axi.py` + `extract_tm.py` + `fig_transmucosal.py` -> `fem_transmucosal.pdf`.
An axisymmetric CAX4 transmucosal section with the Ti implant/abutment, alveolar bone, the peri-implant
mucosa (gingiva cuff, E~3 MPa), the gingival sulcus, and the BIOFILM as a thin layer ON the titanium
WITHIN the sulcus -- i.e. between the Ti and the gingiva, exactly where peri-implant biofilm forms.
Three stages parameterise the pocket bottom: healthy sulcus -> peri-mucositis -> peri-implantitis, where
the biofilm DESCENDS apically along the implant and into the resorbed (infrabony) bone defect (bone
elements there are replaced by biofilm). Confirms the clinical placement; mechanically the soft biofilm/
gingiva change the bone stress little (their role is biological, carried by the inflammation cascade) --
the model's value is the explicit anatomy + the descent path linking sulcus colonisation to the
bone-loss studies.

## Intervention: tooth-brushing / professional debridement
`fig_periimplantitis_brushing.py` -> `fem_periimplantitis_brushing.pdf`. Adds mechanical biofilm removal
to the tipping-point cascade. A toothbrush only reaches ~1-2 mm subgingivally, so its efficacy fades as
the pocket deepens (reachable fraction = clip(1 - L/2mm)); professional debridement reaches the pocket.
For the most-dysbiotic Botelho patient (36-mo bone loss): **no hygiene 2.98 mm; brushing from the start
0.58 mm (controlled); brushing started late (18 mo) 2.94 mm (fails - pocket too deep for the brush);
brushing + professional debridement 0.30 mm (rescued)**. Clinical message: early biofilm control keeps
the system in the stable basin (reversible peri-mucositis), but once the pocket deepens brushing alone
cannot reach the subgingival biofilm -> professional debridement is required. Pure ODE intervention,
no FEM solves.

A polished publication-grade graphical abstract is `fig_graphical_abstract_polished.py` ->
`fem_graphical_abstract_polished.pdf/.png` (numbered stage badges, stage cards, threaded-implant
anatomy schematic).

## Disease-dynamics extensions (5 new figures, ODE/post-processing, no FEM solves)
All five reinforce the cascade with literature-anchored mechanism. Citations are in each script docstring;
anchors summarised under "Literature anchors" below.

1. **Bistability / hysteresis** `fig_periimplantitis_bistability.py` -> `fem_periimplantitis_bistability`.
   Closes the loop the one-way cascade left open: dysbiosis B -> inflammation I, and inflammation FEEDS
   the dysbiotic anaerobes back (bleeding/GCF heme-iron; P. gingivalis is inflammophilic & heme-auxotroph
   -- Hajishengallis keystone-pathogen; FEMS heme review). Mutual Hill activation makes the system
   **bistable**: a healthy and a diseased attractor with a **hysteresis** window (b0 in [0.03, 0.41]).
   Panels: (A) bistable phase plane (two stable fixed points), (B) RANKL/OPG vs plaque burden traced up
   then down -> hysteresis loop ("prevention << cure"), (C) a transient insult locks the patient into the
   diseased basin permanently. This is the deepest upgrade: explains WHY peri-implantitis resists hygiene.
2. **Window of reversibility** `fig_periimplantitis_reversibility.py` -> `..._reversibility`. Heatmap of
   final bone loss over (therapy start time) x (residual dysbiosis), worst-case Botelho patient, with the
   **2 mm peri-implantitis diagnostic contour** separating rescued from lost. Home brushing (residual
   ~0.40, reach-limited) closes its window by ~13-19 mo; professional debridement (residual ~0.15) stays
   sub-threshold even if started ~22 mo. Clinically actionable "how soon / how aggressive".
3. **Host-modulation therapy** `fig_periimplantitis_hostmod.py` -> `..._hostmod`. Same disease attacked at
   three nodes: biofilm control (cuts drive B), anti-cytokine/anti-TNF (raises inflammation resolution),
   anti-RANKL/denosumab (blocks RANKL->osteoclast coupling, downstream). 36-mo loss: none 2.98, anti-TNF
   1.85, anti-RANKL 1.70, biofilm 1.25, **combo 1.20**. Panel B: anti-RANKL arrests bone loss while
   inflammation stays high -- a rescue, not a cure; only biofilm control removes the cause.
4. **Literature calibration & face validity** `fig_periimplantitis_calibration.py` -> `..._calibration`.
   Every rate from literature EXCEPT the single bone-loss gain kL, pinned by least-squares to the
   active-phase peri-implantitis progression anchor (~3 mm @ 36 mo). **Fitted kL = 0.022 wk-1.** The
   calibrated worst-case sits in the peri-implantitis band; the stable site stays inside the healthy
   remodelling envelope. (The non-pathological early-MBL plateau 0.93->1.04 mm is biologic-width
   remodelling -- a different mechanism we deliberately do NOT fit.)
5. **Design buys time** `fig_periimplantitis_design_time.py` -> `..._design_time`. Couples the FEM coupon
   diameter sweep (crestal sigma_p95: O3.5 = 22.2, O4.1 = 14.4, O4.8 = 10.1 MPa) to progression by scaling
   the mechanical-feedback term by crest-stress ratio. Time-to-critical: **O3.5 = 61, O4.1 = 65, O4.8 = 68
   months**. Honest framing: design is a **secondary** modifier (~7 mo over 5+ yr); biofilm control
   dominates -- consistent with overload-needs-inflammation synergy (Chambrone 2010; Naert 2012).

## Literature anchors (for thesis citations; full report in the session log)
- Bone loss: healthy remodelling <=1.5 mm yr1 then <=0.2 mm/yr (Albrektsson 1986; Schwarz 2018 JCP
  doi:10.1111/jcpe.12954); progression in ~64% over ~6.4 yr (PMC9253284); early MBL 0.93/1.04 mm @ 12/36 mo
  (Int J Implant Dent 2025, doi:10.1186/s40729-025-00613-x).
- Mechanostat: 2000/4000 ue remodel/overload thresholds (Frost 2003 Anat Rec 275A:1081); peri-implant FEA
  resorption switch PMC9621276.
- RANKL/OPG elevated in peri-implantitis PICF (Clin Oral Investig 2021, PMID 34264378); OC functional
  lifespan ~2 wk (Parfitt/Jaworski); osteomorph recycling caveat (McDonald 2021 Cell).
- Host modulation: denosumab FREEDOM +5% hip BMD / -68% vertebral fx (Cummings 2009 NEJM 361:756);
  anti-RANKL inhibits alveolar bone destruction (PMID 29607937; Valverde 2025 JPR); anti-TNF reduces
  alveolar bone loss, time-dependent (Kobayashi & Yoshie 2020 Front Immunol 11:591365).
- Ecology<->host feedback: inflammophilic dysbiosis / keystone pathogen (Hajishengallis 2014, PMC4071223);
  P. gingivalis heme-auxotrophy HmuY/gingipains (FEMS Microbiol Rev 2025, doi:10.1093/femsre/fuaf019).
- Overload x biofilm synergy: overload alone insufficient, aggravates only with inflammation (Chambrone
  2010 JP doi:10.1902/jop.2010.100176; Naert 2012 PMID 23062133).
- Bone-remodelling ODE templates: Komarova 2003 Bone 33:206; Lemaire 2004 JTB 229:293 (explicit
  RANK-RANKL-OPG); Pivonka 2008 Bone 43:249. Dysbiosis bistability: Microbiome 2023
  doi:10.1186/s40168-023-01474-5.
- Toothbrush subgingival reach/efficacy collapses at PD>=5 mm (PMC8327450).

## Candidate calibration dataset (NEW, for a future per-timepoint Bayesian fit)
**PRJNA1215005** (Anuntakarun 2025, Int Dent J 75(5):100951, doi:10.1016/j.identj.2025.100951) -- 7 patients,
42 samples, **longitudinal 16S + radiographic bone loss + PD/BOP at baseline/3 mo/6 mo**. Closest open
dataset pairing microbiome with bone-loss over time; the practical longitudinal triad is
Dieckow PRJEB71108 + Botelho PRJNA725874 + PRJNA1215005. No open dataset pairs longitudinal 16S with
longitudinal RANKL/OPG/cytokine in the same subjects (gap flagged).

## Result (honest, root-analog `tier2b_real`)
The fully real-geometry coupled model **solves successfully** — the methodological goal. With a
literature-standard linear PDL (50 MPa) the peri-implant vs peri-tooth crestal-bone contrast is
**modest**: at the crest (z∈[26,28.5]) the osseointegrated implant concentrates slightly more stress
(≈6.6 vs 6.1 MPa) while the PDL-supported tooth distributes load deeper along the root (z∈[16,20]:
2.1 vs 1.6 MPa) — qualitatively the peri-implant marginal-bone-loss signature, but far weaker than the
idealized parametric Tier-2(a). PDL modulus was **not** softened to inflate the contrast.
Suitable as a thesis-Outlook demonstration, not a quantitative claim.
