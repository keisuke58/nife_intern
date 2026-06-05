# Joshi et al. 2025 — Usable Knowledge & Methods for nife Project

DOI: https://doi.org/10.1038/s41522-025-00807-6  
Files: `Szafranski_Published_Work/Joshi2025-Biomarker_Final.pdf`, `..._SI.pdf`

---

## 1. Study in One Sentence

Paired **full-length 16S + metatranscriptomics (RNAseq)** on 48 peri-implant biofilm samples (32 patients, same MHH/NIFE group) identifies a DNA–RNA biomarker panel with AUC=0.85 for peri-implantitis diagnosis.

---

## 2. Cohort Characteristics (Supplementary Files A, B)

### Training cohort (Germany, N=32 patients)

| Variable | Health | Peri-implantitis | p |
|---|---|---|---|
| Age | 68 ± 9 yr | 69 ± 8 yr | ns |
| BOP | 0% | 83% | <0.0001 |
| Pocket depth | 3.2 ± 1.6 mm | 6.9 ± 2.0 mm | <0.0001 |
| Gingival index | 0.3 ± 0.5 | 2.3 ± 0.5 | <0.0001 |
| Suppuration | 0% | 54% | <0.0001 |
| PICF volume | 54 ± 1 µL | 122 ± 44 µL | <0.0001 |
| Total bacterial RNA | 23 ± 75 ng | 169 ± 324 ng | <0.0001 |

### Validation cohort (Italy, external)

| Variable | Health | Peri-implantitis | p |
|---|---|---|---|
| Age | 57 ± 15 yr | 71 ± 11 yr | 0.0002 |
| Pocket depth | 2.2 ± 0.4 mm | 7.3 ± 2.6 mm | <0.0001 |

**Note**: Peri-implantitis samples have 7× higher bacterial RNA yield, which is biologically meaningful (tissue destruction → nutrient release → more biomass).

---

## 3. Community-Level Findings (maps to nife 10-guild taxonomy)

### Class-level shift (full-16S + RNAseq, consistent)

| Guild (nife) | Class | Direction |
|---|---|---|
| Bacilli | *Streptococcus*, *Rothia* | **↑ Health** |
| Actinobacteria | *Actinomyces*, *Rothia* | ↑ Health |
| Fusobacteriia | *Fusobacterium nucleatum* | **↑ Peri-implantitis** |
| Bacteroidia | *Porphyromonas*, *Prevotella*, *Tannerella* | ↑ Peri-implantitis |
| Spirochaetia | *Treponema* | ↑ Peri-implantitis |
| Negativicutes | *Veillonella*, *Selenomonas* | mixed (amino-acid supplier for Fuso in health) |

This is consistent with the Dieckow gLV fit where Bacilli ↔ Fusobacteriia/Bacteroidia are main competitive axes.

### Species strongly enriched in peri-implantitis (FDR-corrected p < 0.2; Suppl. File C excerpts)

| Species | Health mean % | Disease mean % | FDR p | CAP LDA |
|---|---|---|---|---|
| *Fusobacterium nucleatum* | 0.75 | 2.80 | 0.19 | −0.46 |
| *Fusobacterium periodonticum* | 0.044 | 0.005 | 0.11 | +0.42 (health!) |
| *Porphyromonas gingivalis* (various OTUs) | low | higher | variable | negative |
| *Eubacterium brachy* | 0.27 | 0.73 | 0.35 | −0.33 |
| *Fretibacterium sp.* HMT-362 | 0.010 | 0.082 | 0.28 | −0.36 |

### Species enriched in health (CAP LDA > +0.4)

| Species | Health mean % | Disease mean % | FDR p |
|---|---|---|---|
| *Streptococcus salivarius* / HMT-074 | higher | lower | <0.05 |
| *Rothia dentocariosa* | higher | lower | <0.05 |
| *Actinomyces odontolyticus* | 0.27 | 0.074 | 0.19 |
| *Actinomyces sp.* HMT-172 | 0.20 | 0.14 | 0.19 |
| *Eubacterium sulci* | 0.011 | 0.001 | 0.19 |

---

## 4. Functional Layer — Metatranscriptomics (EC-level)

### Key finding: amino acid metabolism is the dominant signal

- Amino acid pathways = most relevant (by EC count) and second most abundant after carbohydrates in both groups.
- **Health**: amino acid *anabolism* (biosynthesis) dominant → nutrient-scarce environment, self-sufficient commensals.
- **Peri-implantitis**: amino acid *catabolism* (utilization) dominant → host tissue degradation releases peptides/AAs as fuel for pathogens.

### Top peri-implantitis-associated enzymatic biomarkers (Random Forest RFE top-8)

| Enzyme | EC (approx.) | Pathway | Taxon (main) |
|---|---|---|---|
| Urocanate hydratase | EC 4.2.1.49 | Histidine catabolism | Fusobacteriia, Bacteroidia |
| Tripeptide aminopeptidase | EC 3.4.11.4 | Peptide utilization | Fusobacteriia |
| Na⁺-transporting NADH:ubiquinone reductase (Na⁺-NQR) | EC 1.6.5.— | Energy metabolism | Fusobacteriia |
| Phosphoenolpyruvate carboxykinase (PEPCK) | EC 4.1.1.49 | Gluconeogenesis / AA entry | Bacteroidia |
| Polyribonucleotide nucleotidyltransferase (PNPase) | EC 2.7.7.8 | RNA turnover | Bacteroidia, Fusobacteriia |

All show FDR p < 0.05 and Cohen's d > 0.8 (large effect), except tripeptide aminopeptidase (trend only).

### KEGG pathways enriched by diagnosis

| Group | Enriched pathways |
|---|---|
| Health | Pyruvate metabolism, galactose, sulfur, starch/sucrose |
| Peri-implantitis | Lipopolysaccharide biosynthesis, histidine/lysine/tryptophan catabolism, butanoate |

Amino acids with elevated catabolism in peri-implantitis: **histidine, lysine, tryptophan** (highest signal); also alanine, aspartate, glutamate, cysteine, methionine, phenylalanine, tyrosine, valine, leucine, isoleucine, proline.

---

## 5. Taxon-Specific Amino Acid Ecology (Fig. 6, 7)

This is the most mechanistically novel part and directly relevant to AGORA cross-feeding priors.

| Class | Strategy | Implication for nife gLV |
|---|---|---|
| Fusobacteriia | **High catabolism, low anabolism** → metabolically dependent on Negativicutes/Bacilli for AA supply in health | Fusobacteriia is an AA *sink*, likely a *cross-feeding receiver* in health |
| Bacteroidia | High catabolism of host-derived protein/peptides | Competitive with Fusobacteriia for host-derived AAs in disease |
| Negativicutes (*Veillonella*, *Selenomonas*) | Supply AAs to Fusobacteriia in health | Positive cross-feeding toward Fusobacteriia |
| Bacilli (*Streptococcus*) | Balanced; auxotrophic suppliers in health | Competition with Fusobacteriia mediated via AA competition |
| Spirochaetia (*Treponema*) | Some convergent strategy with Lancefielda → excess AA substrate in disease |  |

**Konkret nife-relevant**: The Fusobacteriia dependence on Negativicutes for AA supply (cross-feeding) explains why the no-prior Hamilton model finds a *positive* A-matrix element (net cross-feeding) between these guilds — consistent with our permutation test result (p=0.0004 for Hamilton cross-feeding enrichment).

---

## 6. Machine Learning Approach (directly reusable)

### Pipeline

1. **Feature selection**: CAP (Canonical Analysis of Principal Coordinates) constrained ordination per diagnosis. Features with |LDA correlation to CAP1| > 0.4 selected.
2. **Model**: Random Forest classifier, 20× repeated 10-fold CV.
3. **Feature reduction**: Recursive Feature Elimination (RFE) → optimal 8 features.
4. **Validation**: Independent Italian MGX cohort (genus-level only).

### Performance by dataset

| Input | AUC |
|---|---|
| Species + EC combination | **0.85** |
| Species alone | 0.83 |
| Genus + EC | 0.81 |
| EC alone | 0.74 |
| Genus alone | 0.64 |

**Key lesson**: Adding functional (EC) to taxonomic features improves AUC meaningfully; functional alone is weaker. Taxonomy provides presence/absence; transcriptomics provides activity state.

### Statistical tests used

- **PERMANOVA** (Bray-Curtis dissimilarity matrix) for group separation
- **CAP** for supervised ordination; LDA correlation as feature importance
- **Mann-Whitney U** with FDR correction (Benjamini-Hochberg) for individual features
- **Bray-Curtis similarity** for within-vs-between subject variability

---

## 7. Sequencing & Bioinformatics Methods (for Materials/Methods sections)

### Full-length 16S
- **CCS (Circular Consensus Sequencing)** of full-length 16S rRNA gene amplicons → species-level taxonomy
- Reference DB: extended HOMD (eHOMD) with ANI-based species-level TUs
- OTU approach with Prokka + eggNOG annotation for functional EC numbers
- Read mapping: BWA-samse; count: HTSeq htseq-count

### Metatranscriptomics
- rRNA filtered out at annotation step (not depletion at library level)
- Reads mapped to customized human oral metagenome reference (eHOMD-derived TUs present in cohort metagenomes)
- Multi-mapped reads randomly distributed
- EC counts aggregated; multi-EC genes assigned at common hierarchy level

### DNA+RNA co-isolation from low-biomass biofilm
- Enzymatic lysis: lysozyme (2.5 mg/mL) + mutanolysin (50 U/mL) in 10 mM Tris, 1 mM EDTA, pH 8 for 1.5 h at 25°C
- Significantly improves RNA yield vs. standard bead beating alone
- Peri-implantitis samples yield ~7× more bacterial RNA (quantified as total mRNA × fraction bacterial reads)

---

## 8. Connection to joshi_Amatrix_validation.json (already computed)

File: `results/dieckow_cr/joshi_Amatrix_validation.json`

Results using N=95 Dieckow cross-referenced samples:

| Analysis | ρ | p | Interpretation |
|---|---|---|---|
| AGORA net flow vs. Joshi velocity rg | 0.156 | 0.130 | trend, not significant |
| gLV A-matrix vs. Joshi velocity rg | −0.024 | 0.820 | null |
| Fusobacteriia abundance vs. diagnosis | ρ=0.223 | **0.030** | significant; Fuso tracks disease |
| Bray-Curtis vs. diagnosis | −0.065 | 0.531 | null |

**Interpretation**: The Joshi cross-validation shows Fusobacteriia abundance is the strongest guild-level diagnostic marker, consistent with Joshi finding Fn as a top 16S biomarker. The AGORA prior validation is marginal — consistent with our other analyses showing AGORA prior does not reliably improve LOO-CV.

---

## 9. What to cite / reference this paper for in nife manuscript

- **Biological context**: Peri-implant dysbiosis = community shift not single pathogen; Gram+ → Gram- anaerobic transition (cite for introduction)
- **Amino acid metabolism as ecological driver**: The AA cross-feeding ecology (Fusobacteriia as sink, Negativicutes as supplier) supports biological interpretation of positive gLV A-entries between Fusobacteriia ↔ Negativicutes
- **Species-level composition data**: Fn, Pg, Streptococcus, Rothia as reference abundances for validating our guild-level attractors
- **ML approach**: CAP+RF pipeline is a simpler diagnostic complement to our mechanistic gLV approach

---

## 10. Data Available Locally

| File | Content |
|---|---|
| `Szafranski_Published_Work/Joshi2025-Biomarker_Final.pdf` | Main paper (15 pages, extractable text) |
| `Szafranski_Published_Work/Joshi2025-Biomarker_SI.pdf` | SI with Suppl. Files A–J (125 pages) |
| `Datasets/joshi2025_SI/joshi2025_suppC_species.csv` | 414 species: mean_health, mean_periimplantitis, p, fdr_p, lda_cap |
| `Datasets/joshi2025_SI/joshi2025_suppD_genus.csv` | 78 genera: same columns |
| `Datasets/joshi2025_SI/joshi2025_suppE_class.csv` | 19 classes (16S): same columns |
| `Datasets/joshi2025_SI/joshi2025_suppF_EC.csv` | 1584 ECs: metatranscriptome, lda_cap |
| `Datasets/joshi2025_SI/joshi2025_suppG_class_rnaseq.csv` | 22 classes (RNAseq): same columns |
| `Datasets/joshi2025_SI/joshi2025_severity_EC_PD_regression_annotated.csv` | 20 ECs × PD severity regression + direction + pathway_group |
| `Datasets/joshi2025_SI/joshi2025_severity_MDI_formula.csv` | MDI/eMDI formula, features, AUC |
| `Datasets/joshi2025_SI/joshi2025_cross_EC_validation.csv` | 22 ECs; 10 doubly-validated health markers (biomarker LDA AND severity PD regression) |
| `Datasets/szafranski2025_ecopreprint/szafranski2025_community_types.csv` | 4 CTs: diagnosis, guilds, metabolism, nife analogy |
| `Datasets/szafranski2025_ecopreprint/szafranski2025_class_ecological_roles.csv` | 6 classes: top ECs, metabolites, ecological role |
| `Datasets/szafranski2025_ecopreprint/szafranski2025_key_statistics.csv` | 27 PERMANOVA / sample stats from EcoPreprint |
| `results/dieckow_cr/joshi_Amatrix_validation.json` | Cross-validation of gLV A-matrix against Joshi community velocity |
| `results/dieckow_cr/figs/fig_joshi_Amatrix_validation.pdf` | **Key figure**: dysbiosis direction correlation (r=0.686, p=0.041 for RNAseq) |
| `results/dieckow_cr/figs/fig_szafranski_ct_roadmap.pdf` | CT_I→IV roadmap + doubly-validated EC scatter |

---

## 11. Cross-Dataset Validation Summary (2026-06-05)

### Dysbiosis direction correlation (Joshi × Dieckow)

`fig_joshi_Amatrix_validation.py` computes Pearson r between guild-level Δ vectors:

| Comparison | r | p |
|---|---|---|
| Joshi 16S peri−health vs Dieckow CT2−CT1 | 0.525 | 0.147 |
| **Joshi RNAseq peri−health vs Dieckow CT2−CT1** | **0.686** | **0.041** |

**Shared signals**: Bacilli↓ and Bacteroidia↑ are consistent across both datasets.
**Discordant**: Actinobacteria and Betaproteobacteria go in opposite directions — likely Dieckow CT2 is not the most severe state (CT_III/IV in Szafranski terms).

### Szafranski 4-CT → Dieckow mapping

| Szafranski CT | Dieckow analogue | Diagnosis | Key guild |
|---|---|---|---|
| CT_I | CT1 (commensal) | PIH | Bacilli dominant |
| CT_II | — (not captured) | PIM-Neisseria | Betaproteobacteria spike |
| CT_III | between CT1/CT2 | PIM-Bacteroidia/Fuso | Bacteroidia/Fusobacteriia rise |
| CT_IV | CT2 (dysbiotic) | PI-severe | Bacteroidia/Fuso/Spirochaeta |

Dieckow has 10 patients (3 timepoints) → captures primarily CT_I and CT_IV endpoints; the intermediate mucositis states (CT_II/III) are less represented.

### Doubly-validated health-marker ECs

10 ECs confirmed as health markers in **both** Joshi Biomarker (LDA-CAP > 0.4) and Joshi Severity (negative PD regression coefficient):

| EC | Pathway |
|---|---|
| 5.4.99.9 | Galactose metabolism (UDP-galactopyranose mutase) |
| 4.1.2.40 | Galactose metabolism (tagatose-bisphosphate aldolase) |
| 3.4.22.70 | Peptide bond (Sortase A) |
| 3.5.1.5 | Nitrogen / urease |
| 3.2.1.170 | Carbohydrate |
| 3.1.2.20 | Acyl-CoA / lipid |
| 1.7.2.1 | Nitrogen / denitrification |
| 1.1.1.339 | Sugar nucleotide |
| 1.5.1.38 | Oxidoreductase / FMN |
| 2.4.1.109 | Glycosylation / dolichol |

Galactose metabolism is the dominant pathway — consistent with Szafranski EcoPreprint finding galactose ECs (EC 3.2.1.23, EC 1.2.3.3) as CT_I (PIH) hallmarks.
