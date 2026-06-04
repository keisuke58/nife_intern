# Szafrański — papers & data investigation (2026-06-04)

Investigation of **Szymon P. Szafrański** (シモン = NIFE/IKM intern boss, MHH / SFB TRR-298 SIIRI)
publications and the data worth obtaining, oriented to the remaining ~2 months of the
NIFE master's-thesis-direction work. Compiled from a repo scan + web verification
(two agents). Verified items carry DOIs/accessions; unconfirmed items are flagged.

---

## 1. Publications (DOI-verified, relevance-ranked)

| Rel. | Paper | Venue / year | DOI / ID | Value for NIFE |
|---|---|---|---|---|
| ⭐⭐ | **Dieckow** et al. — early implant biofilm, longitudinal 16S | npj Biofilms Microbiomes 10:85, 2024 | 10.1038/s41522-024-00624-3 | **Primary longitudinal dataset (PRJEB71108)**. Its **Supplementary File 1** (microbe–metabolite–enzyme interactions) is the source of the project's **L1/L2 sign priors**. |
| ⭐⭐ | Szafrański et al. — peri-implant ecosystem (127/125 cross-sectional, 4 community types) | bioRxiv preprint → mSystems (in revision) | 10.1101/2025.06.23.661096 | **Student is co-author.** The cross-sectional cohort behind the local 5-genus subset; target for CS/CH/DS/DH vs CT I–IV attractor comparison. |
| ⭐ | **Joshi** et al. — integrative 16S + metatranscriptome diagnostic biomarkers for peri-implantitis | npj Biofilms Microbiomes 11:175, 2025 | 10.1038/s41522-025-00807-6 | **16S + metatranscriptomics (PRJNA1192962)**, 48 samples / 32 patients. Functional/enzyme-level signal → **independent functional validation of the sign priors**. |
| ⭐ | Joshi/Szafrański — submucosal microbiome vs peri-implantitis severity | J Dent Res 2025/26 | 10.1177/00220345251352809 | MDI/eMDI dysbiosis index; severity-correlated genera (49 implants / 34 patients). |
| ⭐ | Szafrański (lead) — periodontitis metatranscriptome (Prevotella, F. nucleatum) | npj Biofilms Microbiomes, ~2017 | PMC5515211 | Older but the **strongest metabolic-interaction (butyrate cross-feeding) content** in his corpus. |
| ○ | organotypic co-culture (S. oralis vs A. actinomycetemcomitans) modulates mucosa | Cell Microbiol, 2019 | 10.1111/cmi.13078 | **In-vitro pairwise interaction = direct sign evidence.** |
| ○ | phage–host network (Aggregatibacter/Haemophilus) | ISME J, 2019 | 10.1038/s41396-019-0450-8 | First author; future phageome layer. |
| ○ | human oral phageome (review) | Periodontol 2000, 2021 | 10.1111/prd.12363 | Future extension. |
| ○ | laser-assisted microbial culturomics | Nat Commun, 2025 | 10.1038/s41467-025-66804-7 | Isolation method; peripheral. |

---

## 2. Datasets & accessions

**Public — download now, no request needed:**
- **PRJEB71108** (ENA) — Dieckow 2024 longitudinal full-length 16S (most valuable for *dynamics*).
- **PRJNA1192962** (NCBI) — Joshi 2025 **16S + metatranscriptome**, 48 samples / 32 patients.

**Already local (this repo):**
- `Datasets/20260416_mSystems_16S_5genera_all_profiles.{txt,xlsx}` — 5-genus × ~127-sample relative abundances (PIH/PIM/PI labels).
- `Datasets/20260416_AbutmentPapernpjBiofilmsDieckow_SI_Relationships.xlsx` (= Suppl. File 1, 351 rows) — microbe–metabolite–enzyme interactions → **L1/L2 prior source**.
- `Szafranski_Published_Work/.../` — preprint + Dieckow + Joshi PDFs.

**Request-only (high value) — extends existing `szafranski_data_plan.md` (S5/S8/metadata requested 2026-04-17, not yet received):**
- **S5**: full 756-species × 125-sample abundance table → per-sample ODE attractor / out-of-sample validation.
- **S8**: class-level EC activity table → validate predicted metabolic fluxes.
- **Per-sample clinical metadata** (patient ID, PPD, BoP, CT assignment) → MDI, covariate adjustment.
- **In-vitro pairwise interaction raw data** (organotypic / co-culture growth & inhibition) → **direct sign-prior validation** (highest leverage).
- **Confirm the preprint's formal accession/DOI & publication status.**

> ⚠️ Correction: a repo note cited `PRJNA119296` — almost certainly a typo for **`PRJNA1192962`**. The preprint's own accession is unverified; confirm with Szafrański.

---

## 3. How Szafrański feeds the model (provenance)

```
Dieckow Suppl. File 1 (351 microbe–metabolite–enzyme triples)
   → build_net_flow_expanded.py  (L1 exp W=2.0/1.5, L2 predicted W=1.0)
   + AGORA2 FBA (L3, W=1.0)        guild_agora_signs.py
   → signed 10-guild interaction prior
   → Hamilton / gLV fit (sign-constrained)   guild_replicator_dieckow.py
   → LOO-CV: gLV α=0.25 best (LOO-RMSE 0.0490, sign-agree 97%)
Szafrański 2025 cohort (5-genus, 127 samples) → out-of-sample attractor / CT validation
```
Central finding: **constrain SIGN only, never magnitude** (FBA flux units ≠ ecological A units; quantitative MacArthur prior fails).

---

## 4. 2-month thesis-direction plan (data-linked)

1. **[no wait] Independent functional validation of the sign priors using the public metatranscriptome (PRJNA1192962 / Joshi 2025 supplements).** Test whether metatranscriptome-expressed metabolic functions support the cross-feeding *directions* encoded in the prior. Strengthens the paper's core claim (priors are data-supported; AGORA isn't). ← *being started via workflow.*
2. **[after S5]** 127-sample out-of-sample ODE attractor vs observed CT validation (scaffolding exists: `dieckow_posterior_predictive.py`, `attractor_analysis.py`).
3. **[after in-vitro data]** Direct sign-agreement test against measured pairwise interactions (currently only AGORA/literature-based).

**Critical path:** send Szafrański a follow-up for S5 / S8 / metadata / in-vitro interaction data / preprint accession (memory policy: email draft only, no send).

---

*Sources: PubMed/PMC/ENA/NCBI (verified DOIs & accessions above) + repo scan (file paths in `ANALYSIS_NOTES.md`, `szafranski_data_plan.md`, `szafranski2025_summary.md`, `build_net_flow_expanded.py`). Unconfirmed items flagged.*
