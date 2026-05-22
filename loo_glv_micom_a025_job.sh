#!/bin/bash
#PBS -N loo_glv_m25
#PBS -l nodes=1:ppn=4
#PBS -l walltime=168:00:00
#PBS -q default
#PBS -j oe
#PBS -o loo_glv_m25_fold${FOLD}_${PBS_JOBID}.fifawc.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# gLV + MICOM τ=0.5 + competition α=0.25 LOO-CV (asymmetric, directed A)
# Untested combination: L1+L2 competition_weight=0.25 + MICOM community FBA L3

cd /home/nishioka/IKM_Hiwi/nife
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu

echo "gLV+MICOM α=0.25 LOO fold ${FOLD} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python run_glv_loo.py \
    --hold ${FOLD} \
    --alpha 0.25 \
    --fit-file fit_glv_hamilton_kegg_expanded_agora_w1p0.json \
    --use-agora \
    --agora-medium micom \
    --micom-fraction 0.5 \
    --tag micom_a025

echo "Finished fold ${FOLD} at $(date)"
