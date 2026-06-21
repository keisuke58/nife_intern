# Implant-FEM literature validation (2026-06-16)

Cross-check of the **implant biomechanics pillar (柱1)** in `FEM_CONDITIONS.md` / `PROJECT_MAP.md`
against the dental-implant FEM + peri-implantitis literature. Goal: confirm the numbers and findings
are defensible, and flag citation/wording fixes for the defence. The biofilm-growth mechanics side is
already covered in `notes/external_resources.md` — this note covers the **implant/bone/crown + clinical
threshold** side that was not yet documented.

**Bottom line:** every headline FEM finding is *supported in direction and rough magnitude* by the
peer-reviewed literature. No result is contradicted. Six small fixes are citation/framing accuracy, not
physics errors. Listed at the end as TODO.

---

## 1. Material properties — all within standard FEM ranges

| Material | Our value (MPa, ν) | Literature range | Verdict | Best citation |
|---|---|---|---|---|
| Ti-6Al-4V | 110000 / 0.34 | 102–136 GPa; Sevimay uses 110 GPa | OK | Geng 2001; Sevimay 2005 |
| Cortical bone | 13700 / 0.30 | 12.6–26.6 GPa (Sevimay 14.8) | OK | Lin 2010; Sevimay 2005 |
| Cancellous (III–IV) | 1000 / 0.30 | 210–1370 (Sevimay D3=1600, D4=690) | OK, **upper edge for type IV** | Sevimay 2005 |
| Dentin | 18000 / 0.31 | 15000–20000 (typ. 18600) | value OK, **citation weak** | Kinney 2003 (not "Lin 2010") |
| PDL | 50 / 0.45 | linear 0.01–1750; 50 is a recognised linear value | value OK, **citation wrong** | **Rees & Jacobsen 1997** |
| Crown e.max | 95000 / 0.30 | 95–105 GPa | OK | consensus |
| Crown zirconia | 210000 | 200–210 GPa | OK | consensus |
| Enamel | 84000 / 0.30 | 70–84 GPa | OK (high end, defensible) | Habelitz RUS |
| Gingiva/mucosa | 3 / 0.45 | FEA 0.88–11; compressive ~1–3.9 | OK | mucosa-FEA lit |

Notes:
- **Cancellous 1000 MPa** sits between Sevimay's D3 (1600) and D4 (690). Defensible for "type III–IV",
  but for low-density posterior (type IV / D4) the conservative value is ~690 MPa. → consider labelling
  as **type III (D3)** or adding a D4≈690 MPa sensitivity case (cancellous stress is sensitive to this).
- **PDL 50 MPa**: the linear 50 MPa value originates from **Rees & Jacobsen 1997** (Biomaterials
  18(14):995–999), who fitted E to match tooth mobility. **Cattaneo 2005 used a NONLINEAR PDL**, so it
  is the right cite for "real PDL is nonlinear" but the WRONG cite for the 50 MPa linear value.

---

## 2. Biomechanical findings — all supported

| Finding | Our claim | Verdict | Literature | Best citation |
|---|---|---|---|---|
| ISO 14801 load | 100 N, 30° oblique, screening | Supported | ISO 14801 specifies 30° worst-case; 100 N is conservative (clinical molar 200–800 N) | ISO 14801:2016; Rito-Macedo 2021 |
| Crown moment arm | crown height ↑ → crestal stress ×1.5 (18→27 MPa) | Supported | crown-height-space drives marginal stress; ~20%/mm; 24→12 mm cut stress ~43% | Ferreira 2021, Int J Implant Dent 7:81 |
| Crown material 2ndary | 21× stiffness → <2% bone stress | Supported | "spongy-bone stress not influenced by crown type"; load path near-rigid | Nokar 2023, Int J Dent |
| Diameter dominates | diameter > length > pitch | Supported | "diameter more important than length"; classic | Qiu 2024 (syst. rev.); Himmlova 2004 |
| Eccentric occlusion | loaded-side crestal ×1.37 | Supported | oblique 111–212% of axial; unilateral cortical peak | Rito-Macedo 2021, JCED 13:e1192 |
| Crestal localisation | peak at neck = where MBL begins | Supported (consensus) | crestal overloading is the established pattern | Qiu 2024; Himmlova 2004 |

Framing fixes:
- **100 N** = standardized/conservative *screening* load, not a peak physiological molar bite (those
  reach 200–800 N). State it that way.
- Attribute the moment-arm effect to **crown height space / moment arm** (Ferreira 2021), not "C/I ratio
  alone" — this is actually the stronger, more precise claim.
- Describe the eccentric peak as on "the loaded/eccentric side" (buccal-vs-lingual depends on load-vector
  convention), not a named anatomical side.

---

## 3. Clinical / threshold claims

| Claim | Our value | Literature | Verdict | Best citation |
|---|---|---|---|---|
| Micromotion threshold | ~150 µm (fibrous above) | tolerated **50–150 µm**; ingrowth ≤28 µm, fibrous ≥150 µm | Supported (150 = upper bound) | Pilliar 1986; Szmukler-Moncler 1998 |
| Computed micromotion | ~7–8 µm (de-integrated, 100 N) | single-digit–low-tens µm for fixed implants; <50 µm "safe" | Supported / reasonable | Kohli 2021 Sci Rep |
| MBL → stress amplif. | 2→8 mm: 14→35 MPa; 2.7→0.7 N/µm | stress rises sharply once loss >1.5 mm; vicious cycle described | Supported (trend; MPa are model outputs) | Materials 2022 15:5866; Vootla/Lin FEA |
| RANKL/OPG ↑ in PI | elevated in disease PICF | RANKL & ratio trend ↑ with severity, BUT 2021 meta: ratio NOT significant pooled | **Partially** — soften wording | Duarte 2009; Rakic 2014; Chaparro meta 2021 |
| TGF-β coupling | resorption→TGF-β→OB recruit; inflammation uncouples | canonical bone biology (matrix TGF-β couples formation to resorption) | Strongly supported | Tang 2009 Nat Med; Crane & Cao 2014 JCI |

Fixes:
- **Micromotion**: cite the threshold as **"50–150 µm (Pilliar 1986; Szmukler-Moncler 1998)"**, a range —
  not a single 150 µm attributed to Brunski (Brunski is a review/secondary cite; Pilliar is primary).
- **RANKL/OPG**: soften "master switch / established biomarker" → it is an **established
  osteoimmunological driver** mechanistically, but a PICF biomarker with **mixed meta-analytic support**
  (the 2021 Clin Oral Investig meta found the *ratio* not statistically significant when pooled). Cite the
  meta-analysis as the honest caveat — strengthens defence credibility.

---

## 4. TODO — citation/wording fixes (apply to FEM_CONDITIONS.md / thesis on approval)

1. PDL E=50 MPa → cite **Rees & Jacobsen 1997**; keep Cattaneo 2005 only for "PDL is nonlinear".
2. Dentin E=18 GPa → verify "Lin 2010" actually lists it; else cite **Kinney et al. 2003**.
3. Cancellous → label **type III (D3)** or add a **D4≈690 MPa** sensitivity case.
4. Micromotion → "**50–150 µm (Pilliar 1986; Szmukler-Moncler 1998)**", a range.
5. 100 N → "standardized/conservative screening load (clinical molar bite 200–800 N)".
6. RANKL/OPG → soften to "driver mechanistically, emerging/contested PICF biomarker (Chaparro 2021 meta)".

---

## 4b. GDI (Guild Dysbiosis Index) driver — examined 2026-06-16

The disease ODE (柱2) is driven by `GDI = log(φ_Bact+φ_Clos+φ_Fuso) − log(φ_Act+φ_Bac+φ_Neg)`,
a class-level log-ratio, via `dL/dt = K·relu(GDI − GDI₀)·A(L)`. Examined whether "driving from GDI"
is defensible.

**Form is sound & citable:** identical structure to the **Microbial Dysbiosis Index** of Gevers et al.
2014 (Cell Host & Microbe 15:382–392) — log(Σ disease taxa) − log(Σ health taxa). Driving disease
dynamics from a dysbiosis log-ratio is standard.

**Guild split — mostly supported, one contested member:**

| Guild (class) | Side | Verdict | Citation |
|---|---|---|---|
| Bacteroidia (Porphyromonas/Tannerella) | DYS | strong (red complex) | Socransky 1998 |
| Bacteroidia ⊃ Prevotella | DYS | debatable (Prevotella mixed) | Abusleme 2013 |
| Fusobacteriia (Fusobacterium) | DYS | supported (orange/bridge) | Socransky 1998; Griffen 2012 |
| Clostridia (Filifactor, Peptostreptococcus) | DYS | supported | Pérez-Chaparro 2014 |
| Actinobacteria (Actinomyces, Rothia) | COM | supported (health) | Abusleme 2013 |
| Bacilli (Streptococcus) | COM | debatable (mixed genus) | Abusleme 2013; Socransky 1998 |
| **Negativicutes (Veillonella)** | COM | **contested** — health-leaning but reported ↑ in periodontitis | Abusleme 2013 (+ contra reports) |

**Biggest weakness:** class-level aggregation discards the within-class health/disease split. Worst
cases: Bacteroidia lumps commensal-ish Prevotella with red-complex Porphyromonas/Tannerella; Bacilli
lumps healthy S. sanguinis/oralis with other streptococci; **Negativicutes/Veillonella on the COMMENSAL
side is the one assignment the literature does not cleanly support**. → Frame GDI as a *coarse
community-state shift*, not species-level pathogen burden. Flag Veillonella explicitly.

**Threshold sensitivity — RUN 2026-06-16** (`extensions/fig_periimplantitis_gdi_threshold.py` →
`figures/fem_periimplantitis_gdi_threshold.pdf`). Compared cohort-relative GDI₀ (40th-percentile =
**−1.31**) vs absolute GDI₀ = 0 on the real Duran-Pinedo periodontitis cohort. **Key result — this is
itself a finding, not just a robustness pass:**
- **Every one of the 15 diseased patients has mean GDI < 0** (range −2.53 … −0.41). So the absolute
  threshold GDI = 0 deactivates the **entire** known-diseased cohort (0/15 progress) — clearly wrong.
- Reason: the class-level GDI is **negatively offset** because the "commensal" classes (Bacilli =
  Streptococcus, Negativicutes = Veillonella, Actinobacteria) numerically dominate the abundance even
  in disease. → **The index has no meaningful absolute zero; only within-cohort relative position is
  interpretable.** This directly reinforces the class-level coarseness critique above.
- The cohort-relative threshold is therefore **necessary** (it re-centres the offset), not just a
  cosmetic choice — but it must be stated as cohort-relative, and absolute GDI values must NOT be
  interpreted (a healthy-cohort-anchored threshold, e.g. Dieckow median, would be the more principled
  absolute anchor — left as a refinement).
- Reassuringly, **risk is rank-driven by GDI, not by the threshold**: predicted 36-mo bone loss vs mean
  GDI has Spearman ρ = **0.97** (p = 3e-9). The threshold only sets the activation floor / how many
  patients are scored as progressing (9/15 under relative), not the ordering.

**Circularity caveat:** GDI drives loss, so "high-GDI patients lose more bone" is near-tautological. The
only thing rescuing it from circularity is the clinical-calibration figure (GDI(t) tracks measured
PD/BOP as therapy resolves disease, Anuntakarun 2025 / PRJNA1215005) — but that cohort has PD only at
baseline + 6 mo (3 mo NaN), so it is **directional, not quantitative**. The per-patient correlation to
clinical severity is **moderate (ρ≈0.46)**, not strong — do not call GDI "clinically validated".

**Naming fix applied:** glossary said "Gingival Dysbiosis Index"; code + outline say "**Guild**" — the
correct name (computed from guilds, not gingival measurements). Fixed in PROJECT_MAP.md.

## 5. Verified citations (full)
- Geng JP, Tan KB, Liu GR. *J Prosthet Dent* 2001;85(6):585–598. (FEA-in-implant-dentistry review)
- Sevimay M, Turhan F, Kılıçarslan MA, Eskitascioglu G. *J Prosthet Dent* 2005;93(3):227–233. (bone-quality FEA; D1–D4 moduli)
- Rees JS, Jacobsen PH. *Biomaterials* 1997;18(14):995–999. (PDL 50 MPa linear)
- Cattaneo PM, Dalstra M, Melsen B. *J Dent Res* 2005;84(5):428–433. (nonlinear PDL)
- Kinney JH, Marshall SJ, Marshall GW. *Crit Rev Oral Biol Med* 2003;14(1):13–29. (dentin/enamel mechanics)
- Himmlová L, Dostálová T, Kácovský A, Konvičková S. *J Prosthet Dent* 2004;91(1):20–25. (diameter > length)
- Qiu Z, et al. *Heliyon* 2024 (PMC10907775). (length/diameter FEA systematic review)
- Ferreira JJ da R, et al. *Int J Implant Dent* 2021;7:81 (PMC8408299). (crown height space → marginal stress)
- Nokar S, et al. *Int J Dent* 2023:1896475 (PMC10735729). (crown material → bone stress negligible)
- Rito-Macedo F, et al. *J Clin Exp Dent* 2021;13(12):e1192 (PMC8715559). (insertion angle/oblique load)
- ISO 14801:2016 — Dynamic loading test for endosseous dental implants. (30° worst-case)
- Pilliar RM, Lee JM, Maniatopoulos C. *Clin Orthop Relat Res* 1986;208:108–113. (micromotion: ≤28 µm ingrowth, ≥150 µm fibrous)
- Szmukler-Moncler S, et al. *J Biomed Mater Res* 1998;43(2):192–203. (50–150 µm tolerated range)
- Kohli N, et al. *Sci Rep* 2021;11:10797. (micromotion systematic review)
- "Effects of Marginal Bone Loss Progression on Stress Distribution — 3D FEA." *Materials* 2022;15(17):5866 (PMC9457366).
- Duarte PM, et al. *Clin Oral Implants Res* 2009;20(5):514–520. (PICF cytokines/RANKL by severity)
- Rakic M, et al. *J Periodontol* 2014;85(11):1566–1574. (sRANKL/OPG prognostic)
- Chaparro et al. *Clin Oral Investig* 2021 (PMID 34264378). (meta-analysis: RANKL/OPG ratio NOT significant pooled — caveat)
- Tang Y, et al. *Nat Med* 2009;15(7):757–765. (TGF-β couples resorption→formation)
- Crane JL, Cao X. *J Clin Invest* 2014;124(2):466–472. (TGF-β coupling review)

**GDI / dysbiosis-index grounding (§4b):**
- Gevers D, et al. *Cell Host & Microbe* 2014;15(3):382–392. (Microbial Dysbiosis Index — log-ratio form)
- Socransky SS, Haffajee AD, Cugini MA, Smith C, Kent RL. *J Clin Periodontol* 1998;25(2):134–144. (subgingival microbial complexes — red/orange)
- Abusleme L, et al. *ISME J* 2013;7(5):1016–1025. (subgingival microbiome health vs periodontitis)
- Griffen AL, et al. *ISME J* 2012;6(6):1176–1185. (periodontitis vs health 16S profiles)
- Pérez-Chaparro PJ, et al. *J Dent Res* 2014;93(9):846–858. (newly identified periodontal pathogens incl. Clostridia members)
- (peri-implantitis taxa, verify author/year) NGS systematic review PMC10668804 — Porphyromonas/Tannerella/Fusobacterium/Filifactor enriched.
