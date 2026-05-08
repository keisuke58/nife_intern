#!/bin/bash
# submit_loo_725874.sh — submit gLV LOO-CV jobs for PRJNA725874
# Run AFTER preprocess_prjna725874.sh (job 40077) finishes
#
# Usage:
#   bash submit_loo_725874.sh             # submit all (alpha=0, 0.25, 0.5) × 15 folds
#   bash submit_loo_725874.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/home/nishioka/IKM_Hiwi/.venv_jax/bin/python"
RUNNER="${SCRIPT_DIR}/run_glv_loo_725874.py"
PHI="${SCRIPT_DIR}/data/prjna725874/phi_guild.npy"
NODE=""   # no node pinning — let PBS schedule across free nodes
PPN=4
ALPHAS=(0.0 0.25 0.5)

DRY_RUN=0
for arg in "$@"; do [[ $arg == --dry-run ]] && DRY_RUN=1; done

# Check phi array exists
if [[ ! -f $PHI ]]; then
    echo "ERROR: phi_guild.npy not found — run preprocess_prjna725874.sh first"
    exit 1
fi

echo "=== gLV LOO-CV PRJNA725874 ==="
echo "  Folds : 0-14 (15 patients)"
echo "  Alphas: ${ALPHAS[*]}"
echo "  Jobs  : $((${#ALPHAS[@]} * 15)) total"
echo ""

submitted=0
for alpha in "${ALPHAS[@]}"; do
    tag="a$(echo $alpha | tr '.' 'p')"
    for fold in $(seq 0 14); do
        out_name="loo_glv_725874_${tag}_fold${fold}"
        job_script=$(mktemp /tmp/loo725_XXXX.sh)
        cat > "$job_script" <<EOF
#!/bin/bash
#PBS -N ${out_name}
#PBS -l nodes=1:ppn=${PPN}
#PBS -j oe
#PBS -o ${SCRIPT_DIR}/results/prjna725874/${out_name}.log

cd ${SCRIPT_DIR}
${PYTHON} ${RUNNER} --hold ${fold} --alpha ${alpha}
EOF
        if [[ $DRY_RUN -eq 1 ]]; then
            echo "[DRY] qsub (alpha=${alpha} fold=${fold})"
        else
            job_id=$(qsub "$job_script")
            echo "  ${out_name}  →  ${job_id}"
            ((submitted++)) || true
        fi
        rm -f "$job_script"
    done
done

[[ $DRY_RUN -eq 0 ]] && echo "" && echo "Total submitted: ${submitted} jobs"
