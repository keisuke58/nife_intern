# nife_intern

Computational oral biofilm pipeline developed during internship at [NIFE](https://nife-hannover.de/en/) (Niedersächsisches Institut für angewandte Zellgewebezüchtung), Hannover — part of the **SIIRI/TRR-298** consortium.

## 日本語（簡易）

このリポジトリは、口腔インプラントバイオフィルムの群集動態を数理モデルで推定・検証するための計算パイプラインです。主論文（投稿準備中）では、縦断16SデータへのgLV/Hamilton ODEフィッティングと、ゲノム規模代謝モデル（AGORA2）を用いた**符号プライア**による正則化が中心テーマです。

```
16S raw reads (ENA/SRA)
  → vsearch (merge→QC→chimera→SILVA classify)
  → 10-guild φ 配列 (N_patients × T × 10)
  → gLV / Hamilton LOO-CV  (+ 代謝的符号プライア)
  → fit_*.json  →  LOO結果  →  図
```

**COMETS/dFBA**（`comets/`）は生態的相互作用の機構論的検証として並走しています。

- 主論文モデル: [`run_glv_loo.py`](./run_glv_loo.py) / [`loo_cv_kegg_prior.py`](./loo_cv_kegg_prior.py)
- COMETS パイプライン: [`comets/run_comets_pipeline.py`](./comets/run_comets_pipeline.py)（Step A/B/C）
- 主な出力: `results/`（JSON, .npy, PDF figure）、`comets/pipeline_results/`

## 用語集

| 用語 | 意味 | 例（ファイル/コマンド） |
|---|---|---|
| gLV | 一般化 Lotka–Volterra モデル。非対称 A 行列、scipy solve_ivp | `guild_replicator_dieckow.py` |
| Hamilton replicator | 対称 A 行列の複製子 ODE（Hamilton ゲーム理論類比）| `loo_cv_kegg_prior.py --model hamilton` |
| NSP | Natural Selection Price ODE（JAX IFT 勾配）| `loo_nsp_ift_v7_gpu.py` |
| LOO-CV | Leave-One-Patient-Out クロスバリデーション | `run_glv_loo.py --hold <fold>` |
| 符号プライア | A 行列要素の符号を代謝証拠で制約するペナルティ項 | `build_net_flow_expanded.py` |
| L1/L2/L3 | 符号プライア層（L1: Szafrański実験的, L2: Szafrański予測, L3: AGORA2 FBA）| `export_sign_prior.py` |
| AGORA2 | ヒト口腔細菌のゲノム規模代謝モデル集（Heinken 2023）| `comets/agora_gems/*.xml` |
| MICOM | コミュニティ FBA（MICOM 実装）の符号プライア | `--agora-medium micom` |
| GEM | Genome-scale metabolic model | `comets/agora_gems/` |
| dFBA | 動的フラックスバランス解析 | `comets/oral_biofilm.py` |
| COMETS | 複数菌種の代謝・増殖シミュレーション枠組み | `comets/run_comets_pipeline.py` |
| guild / φ | 10 class レベル分類群（`GUILD_ORDER` が正準順序）| `guild_replicator_dieckow.py` |
| CS/CH/DS/DH | 4 つの ODE アトラクター（Commensal/Dysbiotic × Static/HOBIC）| `results/` ディレクトリ名 |
| 1D PDE | バイオフィルム深さ方向の反応-拡散方程式 | `glv_pde_1d.py`, `nsp_pde_1d_heine.py` |
| qsub / PBS | クラスタジョブスケジューラ | `*.sh` ジョブスクリプト |

## Overview

Three loosely-coupled modelling pillars share one common contract — the **10-guild class-level taxonomy** defined in `guild_replicator_dieckow.py` (`GUILD_ORDER`). All `.npy` arrays and `fit_*.json` files are positionally indexed by it.

| Pillar | Description | Key scripts |
|--------|-------------|-------------|
| **1. Ecological ODE inference** (paper) | Fit gLV / Hamilton interaction matrix A from longitudinal 16S data, regularised by metabolic sign priors; validate with LOO-CV | `run_glv_loo.py`, `loo_cv_kegg_prior.py`, `build_net_flow_expanded.py` |
| **2. COMETS / dFBA** | Mechanistic 5-species dynamic FBA; cross-validates ecological interactions against AGORA exchange fluxes | `comets/oral_biofilm.py`, `comets/run_comets_pipeline.py` |
| **3. 16S / metadata preprocessing** | Raw reads → guild abundance arrays and sample metadata | `workflow.nf`, `data/metaphlan_pipeline.sh`, `aggregate_dieckow_guilds.py` |

## Repository Structure

```
nife_intern/
├── guild_replicator_dieckow.py   # GUILD_ORDER / replicator ODE / pack-unpack (shared)
├── pub_style.py                  # Shared matplotlib publication style
│
├── build_net_flow_expanded.py    # Construct 3-layer sign prior net-flow matrix
├── export_sign_prior.py          # Export prior to JSON / .npy
├── guild_agora_signs.py          # AGORA2 FBA → sign prior
│
├── run_glv_loo.py                # gLV LOO-CV (CPU, Dieckow)
├── run_glv_loo_725874.py         # gLV LOO-CV (Botelho 7-timepoint)
├── loo_cv_kegg_prior.py          # JAX LOO driver: --model hamilton|glv
├── loo_nsp_ift_v7_gpu.py         # NSP Hamilton LOO-CV (GPU, latest)
├── glv_pde_1d.py                 # gLV 1D reaction-diffusion PDE (depth axis)
├── nsp_pde_1d_heine.py           # NSP 1D spatial PDE (Heine 2025 HOBIC data)
├── run_mdsine2_loo.py            # MDSINE2 Bayesian comparison baseline
│
├── generate_fig127.py            # Paper figures 1, 2, 7
├── generate_fig3456.py           # Paper figures 3–6
├── generate_fig8.py              # Paper figure 8 (LOO A matrix stability)
├── generate_dieckow_paper_figures.py  # Full paper figure batch
│
├── *.sh                          # PBS qsub job scripts (cluster execution)
├── results/                      # Output JSONs, .npy, PDF figures
│   ├── dieckow_fits/             # fit_*.json parameter files
│   ├── dieckow_cr/               # LOO-CV results + NSP comparison
│   ├── glv_pde/                  # 1D PDE depth profiles
│   └── nsp_pde/                  # NSP 1D PDE (CS/CH/DS/DH attractors)
│
├── comets/
│   ├── oral_biofilm.py           # 5-species 0D Monod dFBA
│   ├── spatial_dfba.py           # 2D spatial Monod dFBA (60×40 grid)
│   ├── run_comets_pipeline.py    # Step A (0D) / B (2D) / C (patient-specific)
│   ├── agora_gems/               # AGORA 1.03 SBML models (5 species)
│   ├── notebooks/                # COMETS_beginner.ipynb, visualization
│   └── pipeline_results/         # COMETS output figures and run files
│
├── data/
│   ├── metaphlan_pipeline.sh     # MetaPhlAn 4 PBS pipeline (shotgun NGS)
│   ├── metaphlan_feature_table_to_init_comp.py
│   └── minimap2_16s_pipeline.sh  # Alternative minimap2 16S alignment
│
├── tests/                        # pytest — covers ETL scripts only
├── workflow.nf                   # Nextflow metadata pipeline
├── PAPER_OUTLINE.md              # Manuscript outline (working title, figures, results)
├── ANALYSIS_NOTES.md             # Running research log (Japanese)
└── docs/
    ├── pipeline_overview.md
    ├── loo_results_summary.md
    └── why_micom_worked.md
```

## Guilds (10-class taxonomy, `GUILD_ORDER`)

| Index | Guild | Class / Order (SILVA) |
|-------|-------|-----------------------|
| 0 | Actinobacteria | Actinomycetia |
| 1 | Bacilli | Bacilli |
| 2 | Bacteroidia | Bacteroidia |
| 3 | Betaproteobacteria | Betaproteobacteria |
| 4 | Clostridia | Clostridia, Erysipelotrichia |
| 5 | Coriobacteriia | Coriobacteriia |
| 6 | Fusobacteriia | Fusobacteriia |
| 7 | Gammaproteobacteria | Gammaproteobacteria |
| 8 | Negativicutes | Negativicutes, Veillonellales |
| 9 | Other | — |

The COMETS/dFBA sub-model uses a **5-species subset** (genus level):

| Code | Species | Role |
|------|---------|------|
| So | *Streptococcus oralis* Uo5 | Early colonizer, glucose→lactate |
| An | *Actinomyces naeslundii* str. Howell 279 | Scaffolding, early colonizer |
| Vp | *Veillonella parvula* Te3 DSM 2008 | Obligate lactate cross-feeder |
| Fn | *Fusobacterium nucleatum* ATCC 25586 | Bridge species |
| Pg | *Porphyromonas gingivalis* W83 | Late pathogen, hemin-dependent |

## Datasets

| Dataset | Accession | Shape | Role |
|---------|-----------|-------|------|
| Dieckow 2024 | PRJEB71108 | 10 patients × 3 weeks | Primary longitudinal fit / LOO-CV |
| Botelho 2021 | PRJNA725874 | 15 patients × 7 timepoints | Longer time-series validation |
| Szafrański 2025 | mSystems 16S | 127 cross-sectional samples, 5 genera | Attractor / community-type comparison |
| Heine 2025 | — | in vitro flow-chamber HOBIC | 4 ODE attractors (CS/CH/DS/DH); 1D PDE target |

## Pipeline

### Pillar 1 — Ecological ODE Inference (main paper)

**Sign prior construction** (run once, results cached in `fit_*.json`):

```bash
python build_net_flow_expanded.py   # L1 + L2 (Szafrański) + L3 (AGORA2 FBA)
python export_sign_prior.py         # → data/sign_prior_*.npy
```

**Full fit** (initial parameter estimates, used as warm-start for LOO):

```bash
python loo_cv_kegg_prior.py --model glv --alpha 0.25   # CPU
bash run_hamilton_kegg_gpu.sh                           # Hamilton on GPU (vancouver01)
```

**LOO-CV** (dispatch one fold per PBS job):

```bash
for i in 0 1 2 3 4 5 6 7 8 9; do
  qsub -v FOLD=$i loo_glv_micom_job.sh   # gLV + MICOM sign prior
done
python collect_loo_alpha.py   # aggregate results → comparison table
```

**Key LOO-CV results** (Dieckow 10-fold, predict W2 & W3 from W1):

| Model | α | Sign prior | LOO-RMSE | Sign agreement |
|-------|---|------------|----------|----------------|
| gLV (best) | 0.25 | L1+L2 | **0.0490** | 97% |
| gLV + MICOM | 0.25 | L1+L2+L3 | 0.0516 | **100%** (72/72) |
| Hamilton | 0.00 | L1+L2+AGORA | 0.0562 | — |
| gLV free | — | none | 0.0501 | — |
| NSP (best) | — | — | 0.0906 | — |

**Spatial PDE extension** (1D depth, requires CLSM z-profiles from Heine 2025):

```bash
python glv_pde_1d.py       # gLV reaction-diffusion, 10 guilds, Dieckow bulk φ
python nsp_pde_1d_heine.py # NSP Hamilton, 5 species, 4 HOBIC attractors
```

**MDSINE2 comparison baseline**:

```bash
qsub mdsine2_loo_job.sh    # Bayesian nonparametric gLV (3-timepoint compatible)
```

**Paper figures**:

```bash
python generate_fig127.py              # Figs 1, 2, 7 (study design, AGORA pipeline, A vs F scatter)
python generate_fig3456.py             # Figs 3–6 (sign validation, W-sweep, LOO comparison, A heatmap)
python generate_fig8.py                # Fig 8 (LOO A matrix stability)
python generate_dieckow_paper_figures.py  # Full batch
```

### Pillar 2 — COMETS / dFBA

```bash
# requires nife package alias: ln -s nife_intern nife  (from parent dir)

# Step A: 0D well-mixed healthy vs diseased
python nife/comets/run_comets_pipeline.py --step A

# Step B: 2D spatial (60×40 grid, diffusion)
python nife/comets/run_comets_pipeline.py --step B

# Step C: patient-specific (requires MetaPhlAn init_comp.json)
python nife/comets/run_comets_pipeline.py --step C --init-comp path/to/init_comp.json

# Sobol sensitivity analysis (N=256, 12 Monod params)
qsub comets/run_sobol.sh
```

Key Sobol result: `Fn_mu_max` (ST=0.49) and `Vp_Km_lac` dominate dysbiosis — driven by the lactate cross-feeding bridge, not *Porphyromonas* directly.

See [`comets/AGORA_DFBA.md`](./comets/AGORA_DFBA.md) for the dFBA fallback chain (COMETS Java → AGORA-calibrated Monod → mock logistic).

### Pillar 3 — 16S / Metadata Preprocessing

**16S amplicon** (vsearch + SILVA 138.1):

```bash
qsub data/metaphlan_pipeline.sh            # MetaPhlAn 4 (shotgun NGS, PRJEB71108)
qsub preprocess_prjna725874.sh             # vsearch pipeline (PRJNA725874)
python aggregate_dieckow_guilds.py         # SILVA blast6 → guild φ array
```

**Metadata pipeline** (Nextflow, covers ETL scripts tested by pytest):

```bash
nextflow run workflow.nf
```

**Tests** (metadata/ETL only — run from parent dir with `nife` symlink):

```bash
ln -s nife_intern nife   # one-time alias in parent dir
python -m pytest nife/tests/ -q
python -m pytest nife/tests/test_merge_meta.py -q   # single file
python -m pytest nife/tests/ -k genus_from_taxon    # single test
```

## Key References

- Dieckow et al. 2024, *npj Biofilms Microbiomes* 10:85 — implant biofilm longitudinal data (PRJEB71108)
- Szafrański et al. 2025, *mSystems* — 5-genus cross-sectional data; L1/L2 sign priors
- Heinken et al. 2023, *Nat. Biotechnol.* 41:1320 — AGORA2 genome-scale metabolic models
- Heine et al. 2025, *Frontiers* — HOBIC in vitro flow-chamber (4 attractors, CLSM/FISH data)
- Dukovski et al. 2021, *Nat. Protocols* — COMETS framework
- Frings, Mukherjee et al. 2025, *Analyst* — ATR-FTIR oral bacteria identification
- Joshi et al. 2025, *npj Biofilms Microbiomes* — peri-implantitis submucosal microbiome

## Context

**SIIRI / SFB TRR-298** — Safety Integrated and Infection Reactive Implants  
Group: Prof. Meike Stiesch, MHH Department of Prosthetic Dentistry and Biomedical Materials Science  
Experimental collaborators: Dr. Katharina Szafrański, Dr. Rumjhum Mukherjee, Dr. Pallavi Joshi  
Manuscript status: draft — see [`PAPER_OUTLINE.md`](./PAPER_OUTLINE.md)
