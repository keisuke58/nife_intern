#!/bin/bash
# Batch PBS submission for all 30 Dieckow samples.
# Usage: bash dieckow_batch_submit.sh [--dry-run] [--skip-done]
#
# Submits one PBS job per sample. A_3 is skipped if already done.
# Results → nife/results/dieckow_taxonomy/{sample}_taxonomy.tsv

set -euo pipefail

MANIFEST="/home/nishioka/IKM_Hiwi/nife/dieckow_manifest.tsv"
SCRIPT="/home/nishioka/IKM_Hiwi/nife/dieckow_pbs_job.sh"
OUT_DIR="/home/nishioka/IKM_Hiwi/nife/results/dieckow_taxonomy"
DRY_RUN=0
SKIP_DONE=0

for arg in "$@"; do
    case $arg in
        --dry-run)  DRY_RUN=1 ;;
        --skip-done) SKIP_DONE=1 ;;
    esac
done

# Check free nodes (avoid overloading shared cluster)
echo "=== Cluster status ==="
qstat -u nishioka 2>/dev/null | tail -n +3 | awk '{print $1, $3, $10}' | head -20
echo ""

submitted=0
skipped=0

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r sample patient timepoint err fastq; do
    tsv_out="${OUT_DIR}/${sample}_taxonomy.tsv"

    if [[ $SKIP_DONE -eq 1 && -f "$tsv_out" ]]; then
        echo "[SKIP] ${sample} — already done"
        ((skipped++)) || true
        continue
    fi

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] qsub -v SAMPLE=${sample},ERR=${err} ${SCRIPT}"
    else
        job_id=$(qsub -v SAMPLE="${sample}",ERR="${err}" \
            -l nodes=1:ppn=12 \
            -N "dc_${sample}" \
            "${SCRIPT}")
        echo "[SUBMIT] ${sample} (${err}) → ${job_id}"
        ((submitted++)) || true
        # Small delay to avoid overwhelming the scheduler
        sleep 0.5
    fi
done

if [[ $DRY_RUN -eq 0 ]]; then
    echo ""
    echo "Submitted ${submitted} jobs, skipped ${skipped} (already done)"
    echo "Monitor: watch -n 30 'qstat -u nishioka'"
fi
