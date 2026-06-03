# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A computational research codebase (NIFE internship, SIIRI / SFB TRR-298 consortium) for modelling **oral biofilm community dynamics** relevant to peri-implantitis. It is a *research project oriented toward a manuscript*, not a deployable application: most scripts are run once to produce a number, a fitted-parameter JSON, or a figure. The north-star document is `PAPER_OUTLINE.md` (Hamilton/gLV ODE + genome-scale metabolic sign priors); `ANALYSIS_NOTES.md` is the running research log (largely in Japanese, as is much inline documentation).

There are three loosely-coupled modelling pillars that share one common contract — the **10-guild class-level taxonomy** defined in `guild_replicator_dieckow.py` (`GUILD_ORDER`, `GUILD_COLORS`, `GUILD_SHORT`). Treat that ordering as canonical; many `.npy` arrays and JSON files are positionally indexed by it.

1. **Ecological ODE inference** (the paper) — fit generalized Lotka–Volterra / Hamilton replicator interaction matrices to longitudinal 16S guild abundances, regularized by metabolic *sign priors*, and validate with leave-one-patient-out CV (LOO-CV).
2. **COMETS / dFBA** (`comets/`) — mechanistic genome-scale metabolic simulation of a 5-species community (dynamic FBA), cross-validating the ecological interactions against AGORA metabolic models.
3. **16S / metadata preprocessing** — turn raw sequencing + paper supplements into the guild abundance arrays and metadata tables the models consume (vsearch+SILVA, QIIME2, MetaPhlAn, and a Nextflow metadata pipeline).

## The two import conventions (read before running anything)

The package is referred to throughout as **`nife`**, even though this checkout is named `nife_intern`. On the original cluster the repo lives at `/home/nishioka/IKM_Hiwi/nife` (or `/home/nishioka/nife`), i.e. the directory itself is `nife`. Two incompatible import styles coexist:

- **Runnable scripts** live under `scripts/<category>/` (`fitting/`, `loo_cv/`, `figures/`, `pde/`, `preprocessing/`, `analysis/`). They import siblings *bare*, e.g. `from guild_replicator_dieckow import GUILD_ORDER` or `from pub_style import ...`. → Run these **from the repo root**: `python scripts/loo_cv/run_glv_loo.py --hold 0 ...`. Each script carries a two-line path-shim at the top (`# [nife-pathshim]`) that puts the repo root on `sys.path` so the bare imports resolve regardless of where the file sits, and all `Path(__file__)` anchors point at the repo root (`parents[2]`) so `results/`/`data/` paths are unchanged. Don't remove the shim or "fix" the `parents[2]`.
- **Shared modules stay at repo root** — the ~20 `*.py` that other scripts import (`guild_replicator_dieckow.py`, `build_net_flow_expanded.py`, `pub_style.py`, `loo_cv_kegg_prior.py`, `paper_data.py`, …) plus the ETL modules imported as `nife.X` by `tests/`/`comets/` (`export_model_inputs.py`, `merge_meta.py`, `get_biosample.py`, `extract_supp.py`, `export_qiime2.py`, `run_qiime2_dada2.py`). Keep these at root. A new script that *only* gets run (never imported) belongs in `scripts/<category>/`; a module that other code imports belongs at root.
- **`comets/` package and `tests/`** use the qualified prefix, e.g. `from nife.comets.oral_biofilm import ...`, `from nife import export_model_inputs`. → These require the repo to be importable **as a package named `nife`**, i.e. run from the *parent* directory with the repo dir named `nife` (`python nife/comets/run_comets_pipeline.py ...`, `python -m pytest nife/tests/`).

In this sandbox the directory is `nife_intern`, so package-qualified runs need a name alias first, e.g. from `/home/user`: `ln -s nife_intern nife` then `python -m pytest nife/tests/`.

## Environment & running work

There is **no `requirements.txt`, `setup.py`, or pinned environment** — dependencies are assumed present on the cluster. The scientific stack is: numpy, scipy, pandas, matplotlib, JAX (CPU and GPU), COBRApy + cometspy, PyMC + ArviZ, scikit-learn, SALib, networkx, seaborn; plus the external `vsearch` binary for 16S. A fresh container has **none of these installed** — installation/data download is out of scope unless asked.

Heavy compute does **not** run locally. It is dispatched to an HPC via PBS:

- **PBS batch** — the many `*.sh` files are `qsub` job scripts (`#PBS -l nodes=...`). They `cd` into the hard-coded cluster path and call a specific Python venv (e.g. `.venv_jax/bin/python` for JAX-CPU). LOO folds are submitted as a loop, e.g. `for i in 0..9; do qsub -v FOLD=$i loo_glv_micom_job.sh; done`.
- **GPU jobs** — dispatched over SSH to host `vancouver01`, activating conda env `klempt_fem` (see `run_hamilton_kegg_gpu.sh`).

When editing a job script, keep the absolute cluster paths and venv/conda references intact unless explicitly asked to retarget — they encode where the job actually runs.

### Tests

pytest lives in `tests/` and **only covers the metadata/ETL scripts** (`export_model_inputs`, `export_qiime2`, `extract_supp`, `get_biosample`, `merge_meta`, `run_qiime2_dada2`, plus one integration test). The ODE/dFBA modelling code is **not** unit-tested. Tests import via the `nife.` prefix, so run them as a package:

```bash
# from the parent dir, with the repo importable as `nife`
python -m pytest nife/tests/ -q
python -m pytest nife/tests/test_merge_meta.py -q          # single file
python -m pytest nife/tests/ -k genus_from_taxon           # single test
```

## How the modelling pieces connect

**Data flow** (see `docs/pipeline_overview.md`):

```
raw 16S reads (ENA/SRA)
  → vsearch (merge→QC→chimera→SILVA classify) → guild phi array  (N_patients × T × 10)
  → gLV / Hamilton LOO-CV  (+ metabolic sign prior)
  → fit_*.json  →  LOO results  →  figures
```

- **Fitted-parameter JSON is the interchange format.** A fit script writes `fit_*.json` (the A matrix + per-patient b vectors); LOO scripts read it via `--fit-file`; plotting scripts read the LOO outputs. Don't recompute a fit when a `fit_*.json` already encodes it.
- **Sign priors** are the project's central idea: the off-diagonal *sign* of the interaction matrix A is constrained from metabolic evidence. They are built in layers — L1/L2 from Szafrański's experimental/predicted interactions, L3 from AGORA2 FBA cross-feeding (`build_net_flow_expanded.py`, `export_sign_prior.py`, `guild_agora_signs.py`). The penalty constrains sign, never magnitude (FBA flux units ≠ ecological interaction units — this is a deliberate finding, see `docs/why_micom_worked.md`).
- **Models**: gLV uses an asymmetric A with scipy `solve_ivp` (`guild_replicator_dieckow.py`); Hamilton uses a symmetric A with JAX. Both are *replicator* form (composition sums to 1). `loo_cv_kegg_prior.py` is the consolidated JAX LOO driver (`--model hamilton|glv`).

**Datasets** (cross-referenced in `ANALYSIS_NOTES.md`):

| Name | Accession | Shape | Role |
|------|-----------|-------|------|
| Dieckow 2024 | PRJEB71108 | 10 patients × 3 weeks | primary longitudinal fit/LOO |
| Botelho 2021 | PRJNA725874 | 15 patients × 7 timepoints | longer time-series validation |
| Szafrański 2025 | mSystems 16S | 127 cross-sectional samples, 5 genera | attractor / community-type comparison |
| Heine 2025 | — | in vitro ODE | the four ODE attractors |

The four ODE attractors recur as the codes **CS / CH / DS / DH** (commensal/dysbiotic × static/HOBIC); `results/` run directories and many scripts use these.

### COMETS / dFBA (`comets/`)

`oral_biofilm.py` is a 5-species (So/An/Vp/Fn/Pg) dynamic FBA. Note the fallback chain (documented in `comets/AGORA_DFBA.md`): COMETS Java → **AGORA-calibrated Monod dFBA (primary path)** → mock logistic. Pure FBA returns μ=0 without the VMH diet files, so AGORA GEMs (`comets/agora_gems/*.xml`, AGORA 1.03) are used for *structural* validation while kinetics come from literature Monod parameters. Entry point: `python nife/comets/run_comets_pipeline.py --step A|B|C` (A=0D, B=2D spatial, C=patient-specific from a MetaPhlAn `init_comp.json`). FEM model of a P. gingivalis biofilm is separate: `comets/fem_pg_model.py` (`comets/FEM_PG_MODEL.md`).

### Metadata pipeline (Nextflow)

`workflow.nf` chains `get_biosample.py → merge_meta.py → extract_supp.py` to assemble sample metadata from ENA file reports + paper supplements. These are the scripts the test suite covers. Run with `nextflow run workflow.nf`.

## Conventions

- **Outputs go to `results/<run_name>/`** (timestamped like `Dysbiotic_HOBIC_20260226_040107`, or descriptive like `dieckow_fits/`). COMETS outputs go to `comets/pipeline_results/`. Many result subdirs are committed; large data, DBs, FASTQs, `*.qza/*.qzv`, and PDFs are gitignored (see `.gitignore`).
- **論文用データの正典は `paper_data.py`** (single source of truth). 5-種アトラクター(CS/CH/DS/DH)の論文グレード posterior は **10000-particle TMCMC** = `results/ultimate_10000p/`（DH は `dh_baseline`）。どの run が論文用かを毎回探さず、`from paper_data import paper_5sp_samples, paper_5sp_theta` を使う。論文用 run を差し替えるときは `paper_data.py` の定数だけ直す。`results/` には同状態の探索的な timestamped/`_1000p`/`deeponet_*` run が多数あるが、それらは論文用ではない。
- **Scripts are argparse CLIs** (~60 of them). When adding a runnable script, follow suit and write results as JSON/`.npy` keyed to `GUILD_ORDER`.
- **Plotting**: shared publication style in `pub_style.py`; guild colors/short-labels in `guild_replicator_dieckow.py`. Paper figures are produced by `generate_fig*.py` / `generate_dieckow_paper_figures.py` (vector PDF, Times New Roman).
- **Versioned experiments via filename suffix**, not branches: `_v2`, `_w1p0` (weight=1.0), `_a025` (α=0.25), `loo_nsp_ift_v7_gpu.py`. New variants are typically new files rather than edits to old ones — preserve the old variant unless asked to replace it.
- `intern_doc/` and `Nishioka_Simulations/` are administrative/handoff material (German internship paperwork, a packaged copy of datasets), not part of the analysis pipeline.
- **Keep the repo root lean.** The root deliberately holds only: the ~20 shared `*.py` modules (every one is imported elsewhere — moving any breaks a bare or `nife.` import), the interdependent `*.sh` qsub job scripts (they reference each other by bare name and `cd` to the absolute cluster path — don't relocate without retargeting all of them), `workflow.nf`, and `dieckow_manifest.tsv` (path-referenced by ~6 scripts). Everything else belongs in a subdir: reference papers → `docs/refs/` (gitignored), email drafts → `docs/correspondence/`, internship paperwork → `intern_doc/`. Don't leave loose PDFs/zips/logs at root.
- **Never commit regenerable byproducts.** LaTeX build artifacts (`*.aux/.toc/.nav/.snm/.vrb/.out/.blg/.bbl` …), compiled paper/slide PDFs, and generated `docs/pptx*/` are gitignored. Large local-only data (`HOBIC FISH/` CLSM imaging, `external_data/`) is gitignored too. To clear LaTeX junk physically, target those extensions in the tex dirs — never `git clean -X` (it would wipe the 9 GB imaging data and other ignored-but-precious files).
- **PDF→PPTX tool**: `scripts/figures/pdf_to_pptx.py` (also on PATH as `pdf2pptx`). `--engine image` (default; pixel-perfect, non-editable) or `--engine libreoffice` (editable text via headless `soffice`; needs `libreoffice-impress`/`-draw`).
