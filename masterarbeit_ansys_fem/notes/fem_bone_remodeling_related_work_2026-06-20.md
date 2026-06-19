# Related work — bone-remodeling & dental-implant FEM (standard methods survey, 2026-06-20)

A literature survey of the **standard ("定番") methods** a peri-implant remodeling FEM is expected to
use, assembled to (a) ground the thesis's FEM pillar in the established literature and (b) identify
which standard pieces were already implemented vs. missing. Citations use the keys in
`masterarbeit_ansys_fem/refs.bib` (mirrored into `docs/references.bib`). Confidence flags note where a
single digit should be checked against the primary PDF before final submission. Open-access PDFs for
the modern dental FEAs are in `docs/refs/`.

> **Scope.** This is a draft *related-work* section for the FEM chapter, written for selective reuse
> (per the thesis inclusion policy). It is NOT inserted into the thesis `.tex` — paste/adapt as needed.

---

## 1. Bone apparent density → elastic modulus

Continuum bone FEM maps an apparent (or ash) density field to an isotropic modulus via a power law
`E = a·ρ^b`. The canonical relations:

- **Carter & Hayes 1977** \citep{CarterHayes1977}: the original `E ∝ ρ³` (apparent density). Still the
  most-cited cortical/trabecular continuum law.
- **Morgan, Bayraktar & Keaveny 2003** \citep{Morgan2003}: site-specific apparent-density laws; the
  femoral-neck `E = 6850·ρ^1.49` is the best validated \citep{Schileo2008}.
- **Keyak et al. 1998** \citep{Keyak1998}: the piecewise CT-FEM relation (ash density) used in
  patient-specific femur models; **Keller 1994** \citep{Keller1994} gives `E = 10500·ρ_ash^2.29`.
- Reviews tabulating all of these: **Helgason et al. 2008** \citep{Helgason2008}; conversion
  `ρ_ash/ρ_app ≈ 0.6` \citep{Schileo2008}. Poisson ratio ν = 0.3 is standard for both bone types.

**Patient-specific CT pipeline** (Keyak/Schileo standard): HU → density via a *scanner-specific* linear
phantom calibration `ρ = c·HU + d`, then density → modulus. **Status in this thesis:** the
density→modulus framework is implemented (`bone_standard_laws.py`: `E_carter_hayes`,
`E_morgan_femoral_neck`, `E_keyak_1998`), but the HU→density step is **not** done — it needs
scanner-calibrated CT data we do not have. Our FEM therefore uses region-wise literature moduli
(cortical 13.7 GPa, cancellous 1.0 GPa), the standard approach when patient CT is unavailable.

## 2. Frost mechanostat & adaptive-remodeling theory

The phenomenological basis of load-driven bone adaptation:

- **Frost mechanostat** \citep{Frost1987, Frost2003}: strain-magnitude windows (microstrain) —
  disuse/resorption below **MESr ≈ 50–100 µε**, homeostatic "lazy zone" up to **MESm ≈ 1500 µε**,
  modeling/formation **1500–3000 µε**, **pathological overload > MESp ≈ 3000 µε** (microdamage →
  resorption), fracture ≈ 25 000 µε. Frost stressed these are order-of-magnitude.
- **Huiskes site-specific SED law** \citep{Huiskes1987}: stimulus = strain-energy density per unit mass
  `S = U/ρ`, driven toward a site reference with a symmetric lazy zone.
- **Weinans, Huiskes & Grootenboer 1992** \citep{Weinans1992}: the canonical density-rate ODE
  (formation / lazy-zone / resorption branches; reference `k ≈ 0.004 J/g`). It is mesh-dependent
  ("checkerboard"); the standard fix is the non-local influence function of **Mullender et al. 1994**
  \citep{Mullender1994}. **Huiskes et al. 2000** \citep{Huiskes2000} recast this as a cell-mediated
  surface model where resorption is triggered by *disuse AND microdamage* — the conceptual basis of the
  **dual-threshold** view.

**Key point for peri-implantitis.** The pure SED rule is *monotonic* (more load → more bone). Only the
**upper (overload) threshold** explains why the stress-concentrated implant crest *loses* bone rather
than apposing it. This dual-threshold (lower disuse + upper pathological overload) is what was missing
from the project's earlier remodeling scripts and is now implemented in microstrain units
(`bone_standard_laws.frost_remodeling_sign`, `bone_remodeling_fem_field.py`).

## 3. Dental-implant bone-remodeling FEA (prior applications)

- **Mellal et al. 2004** \citep{Mellal2004}: SED-stimulus remodeling around implants, validated against
  in-vivo data. **Chou, Jagodnik & Müftü 2008** \citep{Chou2008}: Weinans rule at 100 N occlusal load.
  **Eser et al. 2010** \citep{Eser2010}: time-dependent remodeling for immediately-loaded implants
  (Stanford/Carter–Beaupré theory).
- **Gialain et al. 2022** \citep{Gialain2022} (open access): SED bone-remodeling FEA of narrow maxillary
  implants — uses Frost εp = 4000 µε giving overload-resorption SED limits Spc = 109.6 µJ/mm³ (cortical),
  Spt = 10.96 µJ/mm³ (trabecular); highest SED at the buccal crest near the shoulder. The closest
  published analogue to our crestal-overload result.
- **Su et al. 2019** \citep{Su2019} (open access): density-based mandible remodeling parametric study
  (S₀ = 0.008 J/g, BΔt = 2, lazy zone 15 %, ρ ∈ [0.1, 2.2]) — a fully-specified parameter set.
- **Gosai et al. 2024** \citep{Gosai2024} (open access): platform-switched short-implant FEA, crestal
  cortical vM ≈ 29.3 MPa at 120 N/35°, sub-crestal placement reduces stress.

## 4. Clinical grounding & validation practice

- **Most implant-FEA is unvalidated:** only ~9 % of studies validate against any clinical/experimental
  data \citep{Chang2018}. Validation is therefore dominated by **qualitative spatial co-localisation**
  ("peak crestal stress ↔ MBL starts crestally"), not quantitative longitudinal density fits.
- **Marginal bone loss (MBL) magnitudes:** ≈0.1 mm/yr steady-state over 15 yr \citep{Adell1981};
  success criterion **< 0.2 mm/yr after the first year** \citep{Albrektsson1986}; peri-implantitis
  = progressive loss > 2 mm with inflammation.
- **Crestal localisation consensus:** \citep{Oh2002} — breakdown begins at the crest.
- **Overload debate:** animal overload models support overload-driven loss \citep{Isidor1997}, but human
  reviews argue overload is not a *primary* cause without inflammation \citep{Naert2012} — present as an
  open debate. This is precisely the niche of the present work: the **inflammatory RANKL/OPG channel**
  (biofilm-driven) converts mechanical crestal overload into net resorption — the project's vicious-cycle
  thesis.

## 5. Reference crestal stress/strain (sanity band)

- **ISO 14801:2016** \citep{ISO14801_2016} fixes the 30° worst-case loading geometry (the ~100 N
  magnitude is a mastication convention, not prescribed by the standard).
- Reported crestal cortical peak von-Mises under ~100 N oblique loading in normal bone: **≈10–50 MPa**
  (cluster ~20–35) \citep{GengTanLiu2001, Baggi2008, Himmlova2004, Sevimay2005}; Gosai 2024 ≈29 MPa at
  120 N. Peri-implant physiological strain ≈200–2500 µε. **Our model's crestal cortical vM p95
  ≈ 27–29 MPa (22.6 MPa at the C3D4 collar p95) is squarely within this band, ~3.6–4.4× below cortical
  yield** — physiologically plausible and slightly conservative.

## 6. Where this thesis sits

| Standard method | Status |
|---|---|
| Huiskes SED site-specific remodeling | implemented (`bone_remodeling_huiskes.py`) |
| RANKL/OPG mechanistic disease model (Lemaire/Pivonka) | implemented (`fig_periimplantitis_rankl_opg.py`) |
| Overload mechanostat + crestal saucerisation | implemented (`fig_periimplantitis_remodel.py`) |
| Clinical calibration vs longitudinal cohort | implemented (`fig_periimplantitis_clinical_calibration.py`) |
| Density→E laws (Carter-Hayes/Morgan/Keyak) | implemented (`bone_standard_laws.py`) |
| **Frost microstrain windows + dual-threshold** | **added 2026-06-20** (was the gap) |
| HU→E patient-specific mapping | not done — no scanner-calibrated CT data |

The FEM↔clinical link (crestal stress ↔ MBL) is **well-trodden** \citep{GengTanLiu2001, Oh2002} and is
used here as *grounding*, not as a novel claim. The contribution is the **microbiome → inflammation
(RANKL/OPG) → bone-loss → stress → loss vicious cycle** that couples the (standard) mechanical
remodeling to the (project-specific) dysbiosis dynamics.

### Verification flags (check primary PDF before quoting the exact digit)
- Carter–Hayes coefficient 3790 (strain-rate/unit-convention dependent).
- Morgan 2003 per-site a/b for non-femoral-neck sites.
- Weinans k = 0.004 J/g (original) vs. downstream retunes 0.005–0.045.
- Albrektsson 1986 first-year MBL figure (1.0 vs 1.5 mm; the <0.2 mm/yr-after is solid).
- `Talreja2023` entry (peri-implant strain 330–2090 µε) — author/title not yet verified.
- A standalone real-time remodeling-rate constant B does not exist in the literature (always BΔt).
- DEXA/BMD validation of peri-implant FEM does **not** exist — do not claim it.
