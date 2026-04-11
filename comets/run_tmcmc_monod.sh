#!/bin/bash
#PBS -N nife_tmcmc_monod
#PBS -l nodes=1:ppn=12
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/tmcmc_monod.log

set -euo pipefail

CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/nishioka/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metaphlan4

cd /home/nishioka/IKM_Hiwi/nife

echo "[$(date)] Starting TMCMC Monod calibration (diseased) ..."
python comets/run_tmcmc_monod.py \
    --condition diseased \
    --n_particles 500 \
    --n_mcmc_steps 3 \
    --cov_scale 0.4 \
    --noise_cv 0.10 \
    --workers 12 \
    --seed 42 \
    --out comets/pipeline_results

echo "[$(date)] Starting TMCMC Monod calibration (healthy) ..."
python comets/run_tmcmc_monod.py \
    --condition healthy \
    --n_particles 500 \
    --n_mcmc_steps 3 \
    --cov_scale 0.4 \
    --noise_cv 0.10 \
    --workers 12 \
    --seed 42 \
    --out comets/pipeline_results

echo "[$(date)] Done."
