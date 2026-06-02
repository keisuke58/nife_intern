#!/bin/bash
# submit_baseline_loo.sh — submit no-sign-prior baseline LOO (2 models × 10 folds = 20 jobs)
#
# Usage:
#   bash submit_baseline_loo.sh           # submit all
#   bash submit_baseline_loo.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/nishioka/IKM_Hiwi/.venv_jax/bin/python"
HAM_SCRIPT="${SCRIPT_DIR}/scripts/loo_cv/run_hamilton_expanded_loo.py"
GLV_SCRIPT="${SCRIPT_DIR}/scripts/loo_cv/run_glv_loo.py"
FIT_FILE="fit_glv_hamilton_kegg_expanded_agora_w0p5.json"
NODE="frontale04"
PPN=4

DRY_RUN=0
for arg in "$@"; do
    [[ $arg == --dry-run ]] && DRY_RUN=1
done

echo "=== Baseline LOO-CV (no sign prior) ==="
echo "  Models : hamilton, glv"
echo "  Folds  : 0-9 (10 patients)"
echo "  Jobs   : 20"
echo ""

submitted=0
for fold in $(seq 0 9); do

    # Hamilton baseline
    out_name="loo_expanded_noprior_a0p0_fold${fold}"
    job_script=$(mktemp /tmp/base_ham_XXXX.sh)
    cat > "$job_script" <<EOF
#!/bin/bash
#PBS -N ${out_name}
#PBS -l nodes=${NODE}:ppn=${PPN}
#PBS -j oe
#PBS -o ${SCRIPT_DIR}/results/dieckow_cr/${out_name}.log

cd ${SCRIPT_DIR}
${PYTHON} ${HAM_SCRIPT} \\
    --hold ${fold} \\
    --gpu 0 \\
    --alpha 0.0 \\
    --fit-file ${FIT_FILE} \\
    --no-prior
EOF
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] qsub $job_script  (hamilton baseline fold=${fold})"
    else
        job_id=$(qsub "$job_script")
        echo "  submitted ${out_name}  →  ${job_id}"
        ((submitted++)) || true
    fi
    rm -f "$job_script"

    # gLV baseline
    out_name="loo_glv_noprior_a0p0_fold${fold}"
    job_script=$(mktemp /tmp/base_glv_XXXX.sh)
    cat > "$job_script" <<EOF
#!/bin/bash
#PBS -N ${out_name}
#PBS -l nodes=${NODE}:ppn=${PPN}
#PBS -j oe
#PBS -o ${SCRIPT_DIR}/results/dieckow_cr/${out_name}.log

cd ${SCRIPT_DIR}
${PYTHON} ${GLV_SCRIPT} \\
    --hold ${fold} \\
    --alpha 0.0 \\
    --no-prior
EOF
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[DRY] qsub $job_script  (glv baseline fold=${fold})"
    else
        job_id=$(qsub "$job_script")
        echo "  submitted ${out_name}  →  ${job_id}"
        ((submitted++)) || true
    fi
    rm -f "$job_script"

done

[[ $DRY_RUN -eq 0 ]] && echo "" && echo "Total submitted: ${submitted} jobs"
