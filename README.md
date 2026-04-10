# nife_intern

Computational oral biofilm pipeline developed during internship at [NIFE](https://nife-hannover.de/en/) (Niedersächsisches Institut für angewandte Zellgewebezüchtung), Hannover — part of the **SIIRI/TRR-298** consortium.

## Overview

This repository implements an end-to-end computational pipeline connecting patient metagenomic sequencing data to mechanistic biofilm simulation:

```
NGS (shotgun) → MetaPhlAn 4 → init_comp.json → GEM (AGORA) → dFBA → COMETS
```

The pipeline models multi-species oral biofilm on implant surfaces, focusing on the transition between commensal and dysbiotic states relevant to peri-implantitis.

## Repository Structure

```
nife_intern/
├── comets/                    # COMETS / dFBA simulation
│   ├── agora_gems/            # AGORA GEM reconstructions (5 species)
│   ├── notebooks/             # Jupyter notebooks (beginner + visualization)
│   ├── spatial_dfba.py        # 2D spatial Monod dFBA (60×40 grid, 7 species)
│   ├── oral_biofilm.py        # 0D community dFBA model
│   ├── run_comets_pipeline.py # End-to-end pipeline runner (Step A/B/C)
│   ├── sweep_comets_0d.py     # Parameter sweep (glucose, cross-feeding)
│   ├── run_sobol.sh           # Sobol sensitivity analysis (PBS job)
│   ├── make_pipeline_overview.py  # Pipeline figure generator
│   ├── pipeline_overview.tex  # TikZ pipeline diagram
│   └── pipeline_results/      # Output figures and COMETS run files
└── data/
    ├── metaphlan_pipeline.sh           # MetaPhlAn 4 PBS pipeline script
    ├── metaphlan_feature_table_to_init_comp.py  # Profile → init_comp.json
    ├── download_prjeb71108_fastq.py    # FASTQ download helper (PRJEB71108)
    └── PRJEB71108_filereport.tsv       # ENA file report
```

## Species

| Code | Genus | Role |
|------|-------|------|
| Str | *Streptococcus* spp. | Early colonizer, glucose→lactate |
| Act | *Actinomyces / Schaalia* | Scaffolding, early colonizer |
| Vel | *Veillonella* spp. | Obligate lactate cross-feeder (anaerobe) |
| Hae | *Haemophilus parainfluenzae* | Aerobic/facultative, NO₃ reducer |
| Rot | *Rothia* spp. | Health-associated, aerobic |
| Fus | *Fusobacterium* spp. | Bridge species, anaerobe |
| Por | *Porphyromonas* spp. | Late pathogen, deep anaerobe |

## Pipeline

### Step 1 — NGS Profiling (MetaPhlAn 4)

```bash
qsub data/metaphlan_pipeline.sh
```

Runs bowtie2 alignment against the AGORA/CHOCOPhlAn database, produces per-sample taxonomic profiles and converts them to `init_comp.json` (normalized genus fractions for the 7 target genera).

### Step 2 — COMETS Simulation

```bash
# Step A: 0D parameter sweep
python comets/run_comets_pipeline.py --step A

# Step B: 2D spatial healthy vs diseased comparison
python comets/run_comets_pipeline.py --step B

# Step C: Patient-specific (requires MetaPhlAn output)
python comets/run_comets_pipeline.py --step C
```

### Step 3 — Sensitivity Analysis

```bash
qsub comets/run_sobol.sh   # Sobol indices, N=256, 12 params
```

Key result: `Fn_mu_max` (ST=0.49) and `Vp_Km_lac` dominate dysbiosis — driven by lactate cross-feeding bridge, not *Porphyromonas* directly.

## Key References

- Dieckow et al. 2024, *npj Biofilms Microbiomes* 10:155 — implant biofilm ground truth (volume, viability, composition)
- Dukovski et al. 2021, *Nat. Protocols* — COMETS framework
- Frings, Mukherjee et al. 2025, *Analyst* — ATR-FTIR strain-level identification of oral bacteria
- Joshi et al. 2025, *npj Biofilms Microbiomes* — peri-implantitis submucosal microbiome

## Context

**SIIRI / SFB TRR-298** — Safety Integrated and Infection Reactive Implants  
Group: Prof. Meike Stiesch, MHH Department of Prosthetic Dentistry and Biomedical Materials Science  
Experimental collaborators: Dr. Katharina Szafrański, Dr. Rumjhum Mukherjee, Dr. Pallavi Joshi
