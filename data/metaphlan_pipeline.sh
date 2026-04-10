#!/bin/bash
#PBS -N metaphlan_pipeline
#PBS -q default
#PBS -l nodes=frontale01:ppn=24
#PBS -o /home/nishioka/IKM_Hiwi/nife/data/metaphlan_pipeline.log
#PBS -e /home/nishioka/IKM_Hiwi/nife/data/metaphlan_pipeline.err

set -euo pipefail

CONDA_BASE=/home/nishioka/miniconda3
ENV_NAME=metaphlan4
DB_SRC=/home/nishioka/IKM_Hiwi/nife/data/metaphlan_db
DB_USER=/home/nishioka/IKM_Hiwi/nife/data/metaphlan_db_user
DB_NAME=mpa_vJan25_CHOCOPhlAnSGB_202503
FNA="${DB_SRC}/${DB_NAME}.fna"
FASTQ=/home/nishioka/IKM_Hiwi/nife/data/PRJEB71108_fastq/ERR13166576_A_3.fastq.gz
OUTDIR=/home/nishioka/IKM_Hiwi/nife/data/metaphlan_profiles
SAMPLE=ERR13166576_A_3
NCPU=24

echo "[$(date)] === MetaPhlAn pipeline start ==="

# --- 1. Activate conda env (create if needed) ---
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} \|^${ENV_NAME}$"; then
    echo "[$(date)] Creating conda env ${ENV_NAME}..."
    conda create -n "${ENV_NAME}" -c conda-forge -c bioconda \
        metaphlan=4 bowtie2 python=3.10 -y
fi

conda activate "${ENV_NAME}"
echo "[$(date)] Using metaphlan: $(which metaphlan)"
echo "[$(date)] Using bowtie2-build: $(which bowtie2-build)"

# --- 2. Setup user-writable DB dir ---
mkdir -p "${DB_USER}"

# Copy .pkl and _VINFO.csv (world-readable) only if not already there
for f in "${DB_NAME}.pkl" "${DB_NAME}_VINFO.csv"; do
    [ -e "${DB_USER}/${f}" ] || cp "${DB_SRC}/${f}" "${DB_USER}/"
done

# Symlink mpa_latest
ln -sf "${DB_NAME}" "${DB_USER}/mpa_latest" 2>/dev/null || true

# --- 3. Build bowtie2 large index if incomplete ---
REV2="${DB_USER}/${DB_NAME}.rev.2.bt2l"
if [ ! -s "${REV2}" ]; then
    echo "[$(date)] Building bowtie2 large index (this takes ~2-4h)..."
    bowtie2-build \
        --large-index \
        --threads "${NCPU}" \
        "${FNA}" \
        "${DB_USER}/${DB_NAME}"
    echo "[$(date)] Index build complete."
else
    echo "[$(date)] Index already complete, skipping build."
fi

# --- 4. Run MetaPhlAn ---
mkdir -p "${OUTDIR}"
PROFILE="${OUTDIR}/${SAMPLE}.profile.txt"

if [ ! -s "${PROFILE}" ]; then
    echo "[$(date)] Running MetaPhlAn on ${SAMPLE}..."
    metaphlan "${FASTQ}" \
        --input_type fastq \
        --nproc "${NCPU}" \
        --db_dir "${DB_USER}" \
        -x "${DB_NAME}" \
        --mapout "${OUTDIR}/${SAMPLE}.bowtie2.bz2" \
        -o "${PROFILE}"
    echo "[$(date)] MetaPhlAn done: ${PROFILE}"
else
    echo "[$(date)] Profile already exists, skipping MetaPhlAn."
fi

# --- 5. Build feature_table.tsv ---
FEATURE_TABLE="${OUTDIR}/feature_table.tsv"
echo "[$(date)] Building feature_table.tsv..."

# Header: clade_name <tab> sample
echo -e "clade_name\t${SAMPLE}" > "${FEATURE_TABLE}"
# Body: append non-comment lines from profile (clade_name + relative_abundance columns)
grep -v "^#" "${PROFILE}" | awk -F'\t' '{print $1 "\t" $2}' >> "${FEATURE_TABLE}"

# --- 6. Generate init_comp.json ---
INIT_JSON="${OUTDIR}/init_comp_${SAMPLE}.json"
echo "[$(date)] Generating init_comp.json..."
python3 /home/nishioka/IKM_Hiwi/nife/data/metaphlan_feature_table_to_init_comp.py \
    --feature-table "${FEATURE_TABLE}" \
    --sample "${SAMPLE}" \
    --tax-lev s \
    --out "${INIT_JSON}"

echo "[$(date)] === Pipeline complete ==="
echo "Profile:       ${PROFILE}"
echo "Feature table: ${FEATURE_TABLE}"
echo "init_comp:     ${INIT_JSON}"
cat "${INIT_JSON}"
