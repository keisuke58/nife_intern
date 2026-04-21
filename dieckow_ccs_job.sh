#!/bin/bash
#PBS -N dieckow_ccs
#PBS -l nodes=1:ppn=12
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dieckow_ccs_${SAMPLE}.log

# Usage: qsub -v SAMPLE=A_1,ACCESSION=ERR13166574 dieckow_ccs_job.sh
# Processes one Dieckow PacBio fastq through the full CCS pipeline.

FASTQ_DIR="/home/nishioka/IKM_Hiwi/nife/Szafranski_Published_Work/Szafranski_Published_Work/public_data/ENA/PRJEB71108_fastq"
OUT_DIR="/home/nishioka/IKM_Hiwi/nife/results/dieckow_taxonomy"
SCRIPT="/home/nishioka/IKM_Hiwi/nife/dieckow_full_pipeline.py"

export PATH="/home/nishioka/miniconda3/bin:/home/nishioka/miniconda3/envs/metaphlan4/bin:$PATH"

mkdir -p "$OUT_DIR"

FASTQ="${FASTQ_DIR}/${ACCESSION}.fastq.gz"

echo "=== $(date) ==="
echo "Sample: $SAMPLE  Accession: $ACCESSION"
echo "FASTQ: $FASTQ"

if [ ! -f "$FASTQ" ]; then
    echo "ERROR: FASTQ not found: $FASTQ"
    exit 1
fi

python3 "$SCRIPT" \
    --fastq "$FASTQ" \
    --sample "$SAMPLE" \
    --out "$OUT_DIR" \
    --threads 12

echo "=== Done $(date) ==="
