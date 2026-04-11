#!/bin/bash
#PBS -N nife_oed
#PBS -l nodes=1:ppn=1
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/oed.log

set -euo pipefail

CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/nishioka/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metaphlan4

cd /home/nishioka/IKM_Hiwi/nife

echo "[$(date)] Starting OED (80h / diseased) ..."
python comets/run_oed.py \
    --condition diseased \
    --max_cycles 8000 \
    --dt 0.01 \
    --n_obs 6 \
    --n_candidates 40 \
    --noise_cv 0.10 \
    --out comets/pipeline_results

echo "[$(date)] Starting OED (80h / healthy) ..."
python comets/run_oed.py \
    --condition healthy \
    --max_cycles 8000 \
    --dt 0.01 \
    --n_obs 6 \
    --n_candidates 40 \
    --noise_cv 0.10 \
    --out comets/pipeline_results/oed_healthy

echo "[$(date)] Done."
