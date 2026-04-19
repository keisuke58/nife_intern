#!/bin/bash
#PBS -N dieckow_fit_5sp
#PBS -l nodes=1:ppn=12
#PBS -l walltime=24:00:00
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dieckow_fit_5sp.log
#PBS -e /home/nishioka/IKM_Hiwi/nife/results/dieckow_fit_5sp.err

cd /home/nishioka/IKM_Hiwi/nife

source /home/nishioka/.bashrc
conda activate /home/nishioka/IKM_Hiwi/.venv_jax 2>/dev/null || \
    source /home/nishioka/IKM_Hiwi/.venv_jax/bin/activate 2>/dev/null || true

python3 dieckow_hamilton_fit.py \
    --fit \
    --n-particles ${NPART:-500} \
    --sigma ${SIGMA:-0.05} \
    --output results/dieckow_fits/fit_joint_5sp.json

echo "Done: $(date)"
