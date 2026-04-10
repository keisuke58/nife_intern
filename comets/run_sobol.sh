#!/bin/bash
#PBS -N sobol_sensitivity
#PBS -q default
#PBS -l nodes=frontale04:ppn=4
#PBS -o /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/sobol.log
#PBS -e /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/sobol.err

set -euo pipefail
source /home/nishioka/IKM_Hiwi/.venv_jax/bin/activate
cd /home/nishioka/IKM_Hiwi

echo "[$(date)] === Sobol sensitivity N=256 ==="
python nife/comets/sweep_comets_0d.py --sobol-only --sobol-n 256
echo "[$(date)] Done"
ls -lh nife/comets/pipeline_results/sobol_sensitivity.png
