#!/bin/bash
#PBS -N loo_glv_m04
#PBS -l nodes=1:ppn=4
#PBS -l walltime=168:00:00
#PBS -q default
#PBS -j oe
#PBS -o loo_glv_m04_fold${FOLD}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# gLV + MICOM τ=0.4 LOO-CV
#
# Usage:
#   for i in 0 1 2 3 4 5 6 7 8 9; do
#     qsub -v FOLD=$i loo_glv_micom04_job.sh
#   done

cd /home/nishioka/IKM_Hiwi/nife

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu

echo "Starting gLV+MICOM τ=0.4 LOO fold ${FOLD} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python run_glv_loo.py \
    --hold ${FOLD} \
    --alpha 0.0 \
    --fit-file fit_glv_hamilton_kegg_expanded_agora_w1p0.json \
    --use-agora \
    --agora-medium micom \
    --micom-fraction 0.4 \
    --maxiter 2000 \
    --tag micom_t04

echo "Finished fold ${FOLD} at $(date)"
