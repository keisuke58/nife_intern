#!/bin/bash
# Submit PBS jobs for all 30 Dieckow PacBio samples.
# Run from: /home/nishioka/IKM_Hiwi/nife/
# Usage: bash submit_dieckow_ccs_all.sh [--dry-run]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_SCRIPT="$SCRIPT_DIR/dieckow_ccs_job.sh"
RESULTS_DIR="$SCRIPT_DIR/results/dieckow_taxonomy"
DRY=0
[ "$1" = "--dry-run" ] && DRY=1

mkdir -p "$RESULTS_DIR"

declare -A SAMPLES=(
    [ERR13166574]=A_1 [ERR13166575]=A_2 [ERR13166576]=A_3
    [ERR13166577]=B_1 [ERR13166578]=B_2 [ERR13166579]=B_3
    [ERR13166580]=C_1 [ERR13166581]=C_2 [ERR13166582]=C_3
    [ERR13166583]=D_1 [ERR13166584]=D_2 [ERR13166585]=D_3
    [ERR13166586]=E_1 [ERR13166587]=E_2 [ERR13166588]=E_3
    [ERR13166589]=F_1 [ERR13166590]=F_2 [ERR13166591]=F_3
    [ERR13166592]=G_1 [ERR13166593]=G_2 [ERR13166594]=G_3
    [ERR13166595]=H_1 [ERR13166596]=H_2 [ERR13166597]=H_3
    [ERR13166598]=K_1 [ERR13166599]=K_2 [ERR13166600]=K_3
    [ERR13166601]=L_1 [ERR13166602]=L_2 [ERR13166603]=L_3
)

SUBMITTED=0
SKIPPED=0

for ACC in "${!SAMPLES[@]}"; do
    SAMPLE="${SAMPLES[$ACC]}"
    DONE_FILE="$RESULTS_DIR/${SAMPLE}_taxonomy.json"
    LOG_FILE="$SCRIPT_DIR/results/dieckow_ccs_${SAMPLE}.log"

    if [ -f "$DONE_FILE" ]; then
        echo "SKIP $SAMPLE (already done: $DONE_FILE)"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    CMD="qsub -v SAMPLE=${SAMPLE},ACCESSION=${ACC} $JOB_SCRIPT"
    if [ $DRY -eq 1 ]; then
        echo "[DRY] $CMD"
    else
        JID=$(eval $CMD)
        echo "Submitted $SAMPLE ($ACC) → $JID"
    fi
    SUBMITTED=$((SUBMITTED+1))
done

echo ""
echo "Submitted: $SUBMITTED  Skipped (done): $SKIPPED"
