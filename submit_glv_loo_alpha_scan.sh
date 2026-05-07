#!/bin/bash
# submit_glv_loo_alpha_scan.sh — submit gLV LOO-CV alpha scan (3 alphas × 10 folds = 30 jobs)
#
# Usage:
#   bash submit_glv_loo_alpha_scan.sh                  # submit all
#   bash submit_glv_loo_alpha_scan.sh --dry-run        # print commands only
#   bash submit_glv_loo_alpha_scan.sh --alpha=0.5      # single alpha, all 10 folds
#
# Output: nife/results/dieckow_cr/loo_glv_a{alpha}_fold{k}.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/nishioka/IKM_Hiwi/.venv_jax/bin/python"
LOO_SCRIPT="${SCRIPT_DIR}/run_glv_loo.py"
NODE="frontale04"
PPN=4

DRY_RUN=0
ONLY_ALPHA=""
USE_AGORA=0

for arg in "$@"; do
    case $arg in
        --dry-run)   DRY_RUN=1 ;;
        --alpha=*)   ONLY_ALPHA="${arg#*=}" ;;
        --use-agora) USE_AGORA=1 ;;
    esac
done

ALPHAS=("0.0" "0.25" "0.5")
if [[ -n "$ONLY_ALPHA" ]]; then
    ALPHAS=("$ONLY_ALPHA")
fi

echo "=== gLV Alpha scan LOO-CV ==="
echo "  Alphas : ${ALPHAS[*]}"
echo "  Folds  : 0-9 (10 patients)"
echo "  Jobs   : $((${#ALPHAS[@]} * 10))"
echo ""

qstat -u nishioka 2>/dev/null | tail -n +3 | awk '{print "  running:", $1, $3, $10}' | head -10 || true
echo ""

submitted=0
for alpha in "${ALPHAS[@]}"; do
    alpha_tag="a$(echo $alpha | tr '.' 'p')"
    for fold in $(seq 0 9); do
        agora_tag=""
        agora_flag=""
        if [[ $USE_AGORA -eq 1 ]]; then
            agora_tag="_agora"
            agora_flag="--use-agora"
        fi
        out_name="loo_glv${agora_tag}_${alpha_tag}_fold${fold}"

        job_script=$(mktemp /tmp/glv_job_XXXX.sh)
        cat > "$job_script" <<EOF
#!/bin/bash
#PBS -N ${out_name}
#PBS -l nodes=${NODE}:ppn=${PPN}
#PBS -j oe
#PBS -o ${SCRIPT_DIR}/results/dieckow_cr/${out_name}.log

cd ${SCRIPT_DIR}
${PYTHON} ${LOO_SCRIPT} \\
    --hold ${fold} \\
    --alpha ${alpha} \\
    ${agora_flag}
EOF

        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[DRY] qsub $job_script  (alpha=${alpha} fold=${fold})"
        else
            job_id=$(qsub "$job_script")
            echo "  submitted ${out_name}  →  ${job_id}"
            ((submitted++)) || true
        fi
        rm -f "$job_script"
    done
done

[[ $DRY_RUN -eq 0 ]] && echo "" && echo "Total submitted: ${submitted} jobs"
