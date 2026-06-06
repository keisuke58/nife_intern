#!/usr/bin/env bash
#PBS -N joshi_mtx_humann
#PBS -q default
#PBS -l nodes=1:ppn=24
#PBS -l walltime=72:00:00
#PBS -o /home/nishioka/IKM_Hiwi/nife/joshi_mtx_humann.log
#PBS -e /home/nishioka/IKM_Hiwi/nife/joshi_mtx_humann.err

# Joshi 2025 PRJNA1192962 metatranscriptome → MetaPhlAn/HUMAnN → AGORA sign-prior validation.
#
# Outputs:
#   results/joshi_mtx_humann/runinfo.csv
#   results/joshi_mtx_humann/metaphlan/*.profile.txt
#   results/joshi_mtx_humann/humann/<SAMPLE>/*
#   results/joshi_mtx_humann/joined/*
#   results/joshi_mtx_humann_prior_validation/agora_prior_humann_validation.json
#   results/joshi_mtx_humann_prior_validation/agora_prior_humann_edge_support.csv
#
# Typical launch on the cluster:
#   qsub jobs/joshi_mtx_humann_metaphlan_pipeline.sh
#
# Local/dry run examples:
#   NIFE_DIR=$PWD DRY_RUN=1 bash jobs/joshi_mtx_humann_metaphlan_pipeline.sh
#   NIFE_DIR=$PWD MAX_SAMPLES=4 bash jobs/joshi_mtx_humann_metaphlan_pipeline.sh

set -euo pipefail

NIFE_DIR="${NIFE_DIR:-/home/nishioka/IKM_Hiwi/nife}"
CONDA_BASE="${CONDA_BASE:-/home/nishioka/miniconda3}"
ENV_NAME="${ENV_NAME:-joshi_mtx_humann}"
NCPU="${NCPU:-${PBS_NP:-24}}"
PROJECT_ACC="${PROJECT_ACC:-PRJNA1192962}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"          # 0 = all MTX/RNA-Seq samples
DRY_RUN="${DRY_RUN:-0}"

OUT_DIR="${OUT_DIR:-${NIFE_DIR}/results/joshi_mtx_humann}"
SRA_TMP="${SRA_TMP:-/tmp/joshi_mtx_humann_${USER:-nife}_$$}"
RUNINFO="${RUNINFO:-${OUT_DIR}/runinfo.csv}"
METAPHLAN_DIR="${OUT_DIR}/metaphlan"
HUMANN_DIR="${OUT_DIR}/humann"
JOIN_DIR="${OUT_DIR}/joined"
VALIDATION_OUT="${VALIDATION_OUT:-${NIFE_DIR}/results/joshi_mtx_humann_prior_validation}"

mkdir -p "${OUT_DIR}" "${METAPHLAN_DIR}" "${HUMANN_DIR}" "${JOIN_DIR}" "${SRA_TMP}"
cd "${NIFE_DIR}"

echo "[$(date)] === Joshi MTX HUMAnN/MetaPhlAn pipeline start ==="
echo "NIFE_DIR=${NIFE_DIR}"
echo "PROJECT_ACC=${PROJECT_ACC}"
echo "OUT_DIR=${OUT_DIR}"
echo "NCPU=${NCPU}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] Would activate/create conda env ${ENV_NAME}, download ${PROJECT_ACC}, run MetaPhlAn/HUMAnN, and validate prior."
    exit 0
fi

# ---------- 1. Environment ----------
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda.sh not found at ${CONDA_BASE}/etc/profile.d/conda.sh" >&2
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[$(date)] Creating conda env ${ENV_NAME}..."
    conda create -n "${ENV_NAME}" -c conda-forge -c bioconda \
        python=3.10 metaphlan humann sra-tools seqkit bowtie2 samtools -y
fi
conda activate "${ENV_NAME}"

echo "[$(date)] Tools:"
command -v fasterq-dump
command -v metaphlan
command -v humann
command -v humann_join_tables
command -v humann_regroup_table
command -v humann_renorm_table

# ---------- 2. RunInfo: keep metatranscriptome/RNA-Seq libraries ----------
if [[ ! -s "${RUNINFO}" ]]; then
    echo "[$(date)] Fetching SRA RunInfo for ${PROJECT_ACC}..."
    curl -L --retry 5 --retry-delay 10 \
        "https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc=${PROJECT_ACC}" \
        -o "${RUNINFO}"
fi

RUN_LIST="${OUT_DIR}/mtx_runs.tsv"
export RUNINFO RUN_LIST MAX_SAMPLES
python - <<'PY'
import csv, os, re
from pathlib import Path
runinfo = Path(os.environ['RUNINFO'])
out = Path(os.environ['RUN_LIST'])
max_samples = int(os.environ.get('MAX_SAMPLES', '0'))
rows = []
with runinfo.open(newline='') as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        text = ' '.join(str(r.get(k, '')) for k in (
            'LibraryStrategy', 'LibrarySource', 'LibrarySelection', 'Assay_Type',
            'LibraryLayout', 'SampleName', 'Experiment', 'Run'
        )).lower()
        # Joshi PRJNA1192962 contains 16S plus metatranscriptome.  Keep RNA/MTX-like
        # runs and reject amplicon/16S runs.
        if any(x in text for x in ('amplicon', '16s', 'pacbio', 'ccs')):
            continue
        if not any(x in text for x in ('rna-seq', 'transcript', 'metatranscript', 'metagenomic')):
            continue
        run = r.get('Run') or r.get('Run_s')
        sample = r.get('SampleName') or r.get('Sample') or run
        layout = (r.get('LibraryLayout') or 'SINGLE').upper()
        if run:
            rows.append((sample, run, layout))
# de-duplicate run accessions, preserve order
seen = set(); dedup = []
for item in rows:
    if item[1] in seen:
        continue
    seen.add(item[1]); dedup.append(item)
if max_samples > 0:
    dedup = dedup[:max_samples]
out.write_text('\n'.join('\t'.join(x) for x in dedup) + ('\n' if dedup else ''))
print(f"Selected {len(dedup)} RNA/metatranscriptome runs -> {out}")
if not dedup:
    raise SystemExit('No MTX/RNA-Seq runs selected; inspect RunInfo filters.')
PY

# ---------- 3. Per-run MetaPhlAn + HUMAnN ----------
while IFS=$'\t' read -r SAMPLE RUN LAYOUT; do
    [[ -z "${RUN}" ]] && continue
    SAMPLE_SAFE="$(echo "${SAMPLE}_${RUN}" | tr -c 'A-Za-z0-9_.-' '_')"
    SAMPLE_DIR="${SRA_TMP}/${SAMPLE_SAFE}"
    mkdir -p "${SAMPLE_DIR}"
    META_PROFILE="${METAPHLAN_DIR}/${SAMPLE_SAFE}.profile.txt"
    HUMANN_SAMPLE_DIR="${HUMANN_DIR}/${SAMPLE_SAFE}"
    HUMANN_DONE="${HUMANN_SAMPLE_DIR}/${SAMPLE_SAFE}_genefamilies.tsv"

    echo "[$(date)] --- ${SAMPLE_SAFE} (${RUN}, ${LAYOUT}) ---"

    if [[ -s "${HUMANN_DONE}" && -s "${META_PROFILE}" ]]; then
        echo "[$(date)] Skip ${SAMPLE_SAFE}: MetaPhlAn and HUMAnN outputs exist."
        continue
    fi

    prefetch --output-directory "${SAMPLE_DIR}" "${RUN}"
    fasterq-dump --split-files --threads "${NCPU}" --outdir "${SAMPLE_DIR}" "${SAMPLE_DIR}/${RUN}" || \
        fasterq-dump --split-files --threads "${NCPU}" --outdir "${SAMPLE_DIR}" "${RUN}"

    # gzip FASTQ if fasterq-dump produced plain files; seqkit can read both.
    for fq in "${SAMPLE_DIR}"/*.fastq; do
        [[ -e "${fq}" ]] || continue
        gzip -f "${fq}"
    done

    # HUMAnN works with a single input file; concatenate paired reads while
    # retaining per-read records.  This is standard for paired metatranscriptome
    # HUMAnN profiling when insert-pair information is not used by the mapper.
    COMBINED="${SAMPLE_DIR}/${SAMPLE_SAFE}.combined.fastq.gz"
    if compgen -G "${SAMPLE_DIR}/*_1.fastq.gz" > /dev/null && compgen -G "${SAMPLE_DIR}/*_2.fastq.gz" > /dev/null; then
        cat "${SAMPLE_DIR}"/*_1.fastq.gz "${SAMPLE_DIR}"/*_2.fastq.gz > "${COMBINED}"
    else
        cat "${SAMPLE_DIR}"/*.fastq.gz > "${COMBINED}"
    fi

    if [[ ! -s "${META_PROFILE}" ]]; then
        metaphlan "${COMBINED}" \
            --input_type fastq \
            --nproc "${NCPU}" \
            --bowtie2out "${METAPHLAN_DIR}/${SAMPLE_SAFE}.bowtie2.bz2" \
            -o "${META_PROFILE}"
    fi

    if [[ ! -s "${HUMANN_DONE}" ]]; then
        mkdir -p "${HUMANN_SAMPLE_DIR}"
        humann \
            --input "${COMBINED}" \
            --output "${HUMANN_SAMPLE_DIR}" \
            --threads "${NCPU}" \
            --taxonomic-profile "${META_PROFILE}" \
            --output-basename "${SAMPLE_SAFE}"
    fi

    rm -rf "${SAMPLE_DIR}"
done < "${RUN_LIST}"

# ---------- 4. Join + normalize HUMAnN/MetaPhlAn tables ----------
echo "[$(date)] Joining MetaPhlAn profiles..."
merge_metaphlan_tables.py "${METAPHLAN_DIR}"/*.profile.txt > "${JOIN_DIR}/metaphlan_merged.tsv"

echo "[$(date)] Joining HUMAnN gene-family/pathway tables..."
HUMANN_FLAT="${JOIN_DIR}/humann_flat"
rm -rf "${HUMANN_FLAT}"
mkdir -p "${HUMANN_FLAT}"
find "${HUMANN_DIR}" -mindepth 2 -type f \( -name '*_genefamilies.tsv' -o -name '*_pathabundance.tsv' -o -name '*_pathcoverage.tsv' \) \
    -exec ln -sf {} "${HUMANN_FLAT}" \;
humann_join_tables --input "${HUMANN_FLAT}" --file_name genefamilies --output "${JOIN_DIR}/genefamilies.tsv"
humann_join_tables --input "${HUMANN_FLAT}" --file_name pathabundance --output "${JOIN_DIR}/pathabundance.tsv"

humann_renorm_table --input "${JOIN_DIR}/genefamilies.tsv" --output "${JOIN_DIR}/genefamilies_cpm.tsv" --units cpm --update-snames
humann_renorm_table --input "${JOIN_DIR}/pathabundance.tsv" --output "${JOIN_DIR}/pathabundance_relab.tsv" --units relab --update-snames

# EC regrouping is the most direct HUMAnN product for function-level validation.
humann_regroup_table --input "${JOIN_DIR}/genefamilies.tsv" --groups uniref90_level4ec --output "${JOIN_DIR}/ec.tsv"
humann_renorm_table --input "${JOIN_DIR}/ec.tsv" --output "${JOIN_DIR}/ec_relab.tsv" --units relab --update-snames

# Named pathway table improves metabolite keyword matching for the validator.
humann_rename_table --input "${JOIN_DIR}/pathabundance_relab.tsv" --names metacyc-pwy --output "${JOIN_DIR}/pathabundance_relab_named.tsv" || \
    cp "${JOIN_DIR}/pathabundance_relab.tsv" "${JOIN_DIR}/pathabundance_relab_named.tsv"

# ---------- 5. Ensure prior-edge cache exists, then validate ----------
python "${NIFE_DIR}/scripts/analysis/prior_metatx_data.py" --skip-joshi || \
    echo "[$(date)] WARNING: prior edge extraction failed; validator may fail if cache is absent."

python "${NIFE_DIR}/scripts/analysis/validate_agora_prior_humann.py" \
    --humann-table "${JOIN_DIR}/ec_relab.tsv" \
    --humann-table "${JOIN_DIR}/pathabundance_relab_named.tsv" \
    --metaphlan-table "${JOIN_DIR}/metaphlan_merged.tsv" \
    --outdir "${VALIDATION_OUT}"

echo "[$(date)] === Pipeline complete ==="
echo "Joined tables: ${JOIN_DIR}"
echo "Validation:    ${VALIDATION_OUT}"
