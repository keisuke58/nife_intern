#!/bin/bash
#PBS -N dieckow_ccs
#PBS -l nodes=1:ppn=12
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dieckow_ccs_${SAMPLE}.log

# Usage: qsub -v SAMPLE=A_3,ERR=ERR13166576 dieckow_pbs_job.sh
# Or loop: for each sample in manifest

set -euo pipefail

FASTQ_DIR="/home/nishioka/IKM_Hiwi/nife/Szafranski_Published_Work/Szafranski_Published_Work/public_data/ENA/PRJEB71108_fastq"
OUT_DIR="/home/nishioka/IKM_Hiwi/nife/results/dieckow_taxonomy"
CONDA="/home/nishioka/miniconda3/bin/conda"
PYTHON="/home/nishioka/miniconda3/bin/python3"
SCRIPT="/home/nishioka/IKM_Hiwi/nife/dieckow_full_pipeline.py"

mkdir -p "${OUT_DIR}"

echo "[$(date)] Starting ${SAMPLE} (${ERR})"

$PYTHON $SCRIPT \
    --fastq "${FASTQ_DIR}/${ERR}.fastq.gz" \
    --sample "${SAMPLE}" \
    --out "${OUT_DIR}" \
    --threads 12

echo "[$(date)] Done ${SAMPLE}"
