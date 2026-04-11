#!/bin/bash
#PBS -N nife_postpred
#PBS -l nodes=1:ppn=12
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/postpred.log

set -euo pipefail

CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/nishioka/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metaphlan4

cd /home/nishioka/IKM_Hiwi/nife

echo "[$(date)] Starting posterior predictive checks ..."
python comets/run_posterior_predictive.py \
    --out comets/pipeline_results \
    --n_pp_draws 100 \
    --n_cross_draws 200 \
    --workers 12 \
    --seed 0

echo "[$(date)] Done."
