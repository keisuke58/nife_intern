#!/bin/bash
#PBS -N fem_pg_grid
#PBS -l nodes=1:ppn=4
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/fem_pg_grid.log

set -euo pipefail

CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/nishioka/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metaphlan4

cd /home/nishioka/IKM_Hiwi/nife/comets

echo "[$(date)] Starting FEM per-surface grid search ..."
python3 fem_pg_model.py \
    --nx 12 --ny 12 --nz 8 \
    --days 6 --dt_h 1.0 \
    --fit_fig2 \
    --out /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/fem_pg_fit_fig2_v7.png

echo "[$(date)] Done."
