#!/bin/bash
#PBS -N loo_combined
#PBS -l nodes=1:ppn=4
#PBS -l walltime=168:00:00
#PBS -q default
#PBS -j oe
#PBS -o loo_combined_g${GAMMA}_fold${FOLD}_${PBS_JOBID}.fifawc.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# Hamilton + combined prior (sign + OLS-calibrated magnitude) LOO-CV
#
# Usage:
#   for fold in 0 1 2 3 4 5 6 7 8 9; do
#     for gamma in 0.25 0.50 1.00; do
#       qsub -v FOLD=$fold,GAMMA=$gamma loo_combined_job.sh
#     done
#   done
#
# GAMMA=0.0 → pure sign prior (same as agora_w1p0 baseline)
# GAMMA=0.5 → equal-weight combined
# GAMMA=1.0 → magnitude as strong as sign

GAMMA=${GAMMA:-0.50}

cd /home/nishioka/IKM_Hiwi/nife
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu

echo "Hamilton combined prior LOO fold=${FOLD} gamma=${GAMMA} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python scripts/loo_cv/run_hamilton_expanded_loo.py \
    --hold ${FOLD} \
    --gpu 0 \
    --use-agora \
    --prior-mode combined \
    --gamma-mag ${GAMMA} \
    --sigma-mag 0.3 \
    --fit-file fit_glv_hamilton_kegg_expanded_agora_w1p0.json \
    --tag combined

echo "Finished fold ${FOLD} gamma=${GAMMA} at $(date)"
