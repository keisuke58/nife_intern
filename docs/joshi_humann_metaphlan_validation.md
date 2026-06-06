# Joshi metatranscriptome HUMAnN/MetaPhlAn validation

Purpose: use the public Joshi et al. 2025 metatranscriptome (PRJNA1192962) as an independent functional check that the AGORA/Szafrański sign prior is biologically expressed in peri-implant samples.

## Workflow

1. Fetch NCBI SRA `RunInfo` for `PRJNA1192962`.
2. Keep RNA/metatranscriptome-like libraries and exclude 16S/amplicon/PacBio runs.
3. For each selected run:
   - download reads with `prefetch`/`fasterq-dump`;
   - profile taxa with MetaPhlAn;
   - profile transcript functions with HUMAnN using the MetaPhlAn profile.
4. Join HUMAnN tables, regroup gene families to EC numbers, normalize to relative abundance, and keep a named MetaCyc pathway table for metabolite-keyword matching.
5. Run `scripts/analysis/validate_agora_prior_humann.py` to score whether prior edges are functionally supported.

## Cluster command

```bash
qsub jobs/joshi_mtx_humann_metaphlan_pipeline.sh
```

Useful bounded test:

```bash
NIFE_DIR=$PWD MAX_SAMPLES=4 bash jobs/joshi_mtx_humann_metaphlan_pipeline.sh
```

Dry-run without downloads or conda setup:

```bash
NIFE_DIR=$PWD DRY_RUN=1 bash jobs/joshi_mtx_humann_metaphlan_pipeline.sh
```

## Validator inputs and outputs

The validator can also be run manually once HUMAnN/MetaPhlAn joined tables exist:

```bash
python scripts/analysis/validate_agora_prior_humann.py \
  --humann-table results/joshi_mtx_humann/joined/ec_relab.tsv \
  --humann-table results/joshi_mtx_humann/joined/pathabundance_relab_named.tsv \
  --metaphlan-table results/joshi_mtx_humann/joined/metaphlan_merged.tsv \
  --outdir results/joshi_mtx_humann_prior_validation
```

Outputs:

- `results/joshi_mtx_humann_prior_validation/agora_prior_humann_validation.json`
- `results/joshi_mtx_humann_prior_validation/agora_prior_humann_edge_support.csv`

## Interpretation

An edge is considered testable only when the edge metabolite has a configured function/pathway marker in the HUMAnN output and both endpoint guilds are represented by HUMAnN or MetaPhlAn. Cross-feeding and competition edges are counted as supported when both endpoint guilds show metabolite-associated functional activity. Inhibition edges are counted as supported when the producer shows marker activity and the target guild is taxonomically present.

The default marker map is intentionally broad for exploration. For manuscript-grade results, freeze a curated JSON marker file and pass it with `--metabolite-map`.
