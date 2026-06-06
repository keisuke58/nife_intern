# nife

[![CI](https://github.com/keisuke58/nife_intern/actions/workflows/ci.yml/badge.svg)](https://github.com/keisuke58/nife_intern/actions/workflows/ci.yml)

Computational oral biofilm pipeline developed during internship at [NIFE](https://nife-hannover.de/en/) (Niedersächsisches Institut für angewandte Zellgewebezüchtung), Hannover — part of the **SIIRI/TRR-298** consortium.

**North star:** Hamilton/gLV ecological inference with metabolic sign priors → see `PAPER_OUTLINE.md`.

## 日本語（簡易）

このリポジトリは、患者メタゲノム（ショットガンNGS）の菌叢データから、口腔バイオフィルムの代謝モデル（dFBA/COMETS）までを一気通貫でつなぐ計算パイプラインです。

```
NGS（shotgun）→ MetaPhlAn 4（菌叢プロファイル）→ init_comp.json（属レベル割合）
→ GEM（AGORAの代謝モデル）→ dFBA → COMETS（時間発展シミュレーション）
```

- まず試す: [COMETS_beginner.ipynb](./comets/notebooks/COMETS_beginner.ipynb)（最低限の動作・可視化）
- パイプライン実行: [run_comets_pipeline.py](./comets/run_comets_pipeline.py)（Step A/B/C）
- 菌叢側（MetaPhlAn）: [metaphlan_pipeline.sh](./data/metaphlan_pipeline.sh) → [metaphlan_feature_table_to_init_comp.py](./data/metaphlan_feature_table_to_init_comp.py)
- 主な出力: `comets/pipeline_results/`（図、COMETSの入出力、比較結果）

## 用語集（簡易）

| 用語 | 意味（このrepoでの使い方） | 例（ファイル/コマンド） |
|---|---|---|
| NGS（shotgun） | メタゲノムのショットガンシーケンス。生データから菌叢組成を推定する入口 | `data/` |
| MetaPhlAn 4 | リードから分類学的プロファイル（菌種/属の相対存在量）を推定 | `qsub data/metaphlan_pipeline.sh` |
| taxonomic profile | サンプルごとの菌叢組成（相対存在量の表） | MetaPhlAn出力 |
| init_comp.json | COMETS側の初期組成。対象7属の割合に正規化したJSON | `data/metaphlan_feature_table_to_init_comp.py` |
| GEM | Genome-scale metabolic model（ゲノム規模代謝モデル） | `comets/agora_gems/` |
| AGORA | ヒト腸内細菌などのGEMコレクション。ここでは口腔細菌GEMを利用 | `comets/agora_gems/*.xml` |
| dFBA | 動的フラックスバランス解析。代謝（FBA）と環境（基質）の時間変化を結合 | `comets/oral_biofilm.py` |
| COMETS | 複数菌種の代謝・増殖を（空間あり/なしで）シミュレートする枠組み | `comets/run_comets_pipeline.py` |
| 0D / 2D | 0Dは空間なし（混合）。2Dは格子上で空間あり（拡散など） | Step A（0D）, Step B（2D） |
| cross-feeding | ある菌が作った代謝産物を別の菌が利用する現象 | 乳酸（lactate）など |
| Sobol感度解析 | パラメータの不確実性が結果へ与える寄与（全効果STなど）を推定 | `qsub comets/run_sobol.sh` |
| qsub / PBS | クラスタ投入（ジョブスケジューラ） | `qsub ...` |

## Overview

This repository implements an end-to-end computational pipeline connecting patient metagenomic sequencing data to mechanistic biofilm simulation:

```
NGS (shotgun) → MetaPhlAn 4 → init_comp.json → GEM (AGORA) → dFBA → COMETS
```

The pipeline models multi-species oral biofilm on implant surfaces, focusing on the transition between commensal and dysbiotic states relevant to peri-implantitis.

## Quick Start

```bash
pip install -e ".[dev]"
./scripts/reproduce_core.sh   # LIGHT figures + ETL tests (no cluster)
make figures                  # matplotlib data figures only
pytest tests/ -q              # metadata / QIIME2 ETL tests
```

## Repository Structure

```
nife/
├── paper_data.py              # Canonical paper-grade run paths (single source of truth)
├── guild_replicator_dieckow.py  # 10-guild taxonomy + gLV ODE
├── scripts/
│   ├── fitting/               # gLV / Hamilton fits
│   ├── loo_cv/                # Leave-one-patient-out validation
│   ├── figures/               # Thesis + paper figure generators
│   ├── pde/                   # Spatial diffusion / PINN
│   └── analysis/              # ML extensions (symbolic regression, attractor predictor)
├── dieckow_paper/             # Manuscript figures (make_figures.py)
├── comets/                    # COMETS / dFBA simulation (5-species mechanistic)
├── jobs/                      # PBS / GPU job scripts (cluster only)
├── tests/                     # ETL unit tests (pytest)
├── docs/ARCHITECTURE.md       # Data-flow diagram
├── docs/PROVENANCE.md         # Per-figure regeneration commands
├── Makefile                   # `make figures` (LIGHT only)
└── data/                      # NGS preprocessing scripts
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

## Documentation

| Doc | Purpose |
|-----|---------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Core data flow + three pillars |
| [docs/PROVENANCE.md](docs/PROVENANCE.md) | Figure regeneration (LIGHT vs HEAVY) |
| [docs/ZENODO.md](docs/ZENODO.md) | Zenodo archive + citation |
| [AGENTS.md](AGENTS.md) | AI agent / slash-command entry points |
| [CLAUDE.md](CLAUDE.md) | Full developer guide |
| [PAPER_OUTLINE.md](PAPER_OUTLINE.md) | Manuscript structure |

## Citation

See [CITATION.cff](CITATION.cff) and [docs/ZENODO.md](docs/ZENODO.md). Interim BibTeX:

```bibtex
@software{nishioka2026nife,
  author  = {Nishioka, Keisuke},
  title   = {nife},
  year    = {2026},
  url     = {https://github.com/keisuke58/nife_intern},
  version = {0.2.0}
}
```

## Context

**SIIRI / SFB TRR-298** — Safety Integrated and Infection Reactive Implants  
Group: Prof. Meike Stiesch, MHH Department of Prosthetic Dentistry and Biomedical Materials Science  
Experimental collaborators: Dr. Katharina Szafrański, Dr. Rumjhum Mukherjee, Dr. Pallavi Joshi
