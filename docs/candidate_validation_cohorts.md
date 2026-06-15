# Candidate cross-cohort validation datasets (oral 16S, longitudinal)

Purpose: find additional **public longitudinal oral 16S cohorts** to extend the
cross-cohort gLV/Hamilton validation beyond Dieckow (PRJEB71108) and Duran-Pinedo
2021 (PRJNA725874). Selection: longitudinal (≥3 timepoints ideal), raw reads on
ENA/SRA (so `vsearch+SILVA → 10-guild class-level phi array` works), oral
(subgingival/submucosal best = matches Dieckow). Compiled 2026-06-16.

> Rule (this project burned a citation error once): **never write an accession we
> have not verified.** Accessions below are tagged with how they were checked.

## Ranked shortlist

| # | Accession | Design (subjects × timepoints) | Site / condition | 16S | Status |
|---|-----------|-------------------------------|------------------|-----|--------|
| **1** | **PRJNA1215005** | 7 × 3 (baseline / 3mo / 6mo), healthy+diseased paired | subgingival, **peri-implantitis** | V3–V4 | ✅ stated in paper PMC12361765 (verbatim). ⚠ confirm SRA release (esearch returned 0 runs — may be embargoed/just released). Already named in `masterarbeit_ansys_fem/.../README`. **Best fully-actionable match.** |
| **2** | *Vílchez 2026* — **accession TBD** | 27 / 64 implants × 3 (**1wk / 4wk / 3yr**) | submucosal, **peri-implant** | 16S | ⚠ accession is in the **paywalled** Data Availability Statement (not on EPMC/PMC). Clin Oral Implants Res 2026;37:558–574, doi:10.1111/clr.70101. **Best design match** (1wk/4wk ≈ Dieckow early colonization). Get accession via institutional access. |
| 3 | **PRJNA255922** | 12 × 2 (pre / post treatment; 3 subj a 3rd) | subgingival, periodontitis | — | ✅ verified NCBI ("Oral microbiome Metagenome", 95 runs). ⚠ only 2 tp; **same Frías-López group as Duran-Pinedo → not independent**. Low priority. |
| 4 | **PRJNA786436** | paired responder / non-responder sites, pre/post | subgingival, periodontitis | — | ✅ verified NCBI. Only ~2 tp. |
| — | **PRJNA622300** | 56 children × 5 (6-mo intervals, 24mo) | saliva, ECC | V-region | ✅ verified NCBI ("Predicting ECC Risk…"). Different ecosystem (saliva/children); robustness only. |
| — | **SRP040945 / SRP040947** | 50 preschoolers × multi-tp / 2yr | plaque + saliva, ECC | V4 | ✅ exists in SRA (Teng 2015, Cell Host & Microbe). Different ecosystem. |
| — | COHRA2 | 189 children, 855 samples | saliva, ECC | 16S | large, case-control sampling; accession not yet pulled. |

## TODO — tomorrow on the school/university wifi (2026-06-17)

1. **Vílchez accession** — open the paper via **MHH / LUH institutional Wiley access**
   (https://onlinelibrary.wiley.com/doi/10.1111/clr.70101), read the *Data
   Availability Statement*, copy the BioProject/SRA accession here. (Or email the
   corresponding author.) Do **not** guess it.
2. **PRJNA1215005** — check it is actually downloadable:
   - ENA: https://www.ebi.ac.uk/ena/browser/view/PRJNA1215005 (look for FASTQ run links)
   - or SRA: `prefetch PRJNA1215005` / check on https://www.ncbi.nlm.nih.gov/sra/?term=PRJNA1215005
   - if embargoed, note the release date.
3. If reads are available → mirror the Duran-Pinedo path: raw reads → `vsearch+SILVA`
   → `data/<accession>/phi_guild.npy` (N×T×10, `GUILD_ORDER`), analogous to
   `data/prjna725874/`. Template: `scripts/preprocessing/build_guild_phi_725874.py`.
4. Then it drops straight into the existing prior-free gLV cross-cohort comparison
   (`fit_duranpinedo_A.py` analogue) and the verification battery
   (`scripts/analysis/verify_*.py`).

## Why this matters
PRJNA1215005 / Vílchez would be the **first independent cohort that matches Dieckow's
site and condition** (peri-implant, early-to-mid longitudinal), unlike Duran-Pinedo
(periodontitis, slow progression) — directly addressing Szafranski's design-gap
concern (see `docs/cross_cohort_verification_2026-06-15.md`).

## Sources
- Peri-implantitis longitudinal (PRJNA1215005): https://pmc.ncbi.nlm.nih.gov/articles/PMC12361765/
- Vílchez 3-yr peri-implant: https://onlinelibrary.wiley.com/doi/10.1111/clr.70101
- Yost mBio (PRJNA255922): https://pmc.ncbi.nlm.nih.gov/articles/PMC4337560/
- PRJNA786436: https://pmc.ncbi.nlm.nih.gov/articles/PMC8920355/
- ECC PRJNA622300: https://pmc.ncbi.nlm.nih.gov/articles/PMC8142088/
- Teng 2015 ECC: https://www.cell.com/cell-host-microbe/fulltext/S1931-3128(15)00333-9
