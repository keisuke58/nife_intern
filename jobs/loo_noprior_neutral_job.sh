#!/bin/bash
#PBS -N loo_noprior_neutral
#PBS -l nodes=1:ppn=4
#PBS -l walltime=48:00:00
#PBS -q default
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dieckow_cr/loo_noprior_neutral_fold${FOLD}.log

# Neutral-init no-prior LOO: start from zero A, no sign penalty
# Purpose: gold-standard test of warm-start contamination for sign enrichment

cd /home/nishioka/IKM_Hiwi/nife

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=4"

echo "neutral-init no-prior LOO fold ${FOLD} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python scripts/loo_cv/run_hamilton_expanded_loo.py \
    --hold ${FOLD} \
    --gpu -1 \
    --fit-file results/dieckow_cr/fit_zero_init.json \
    --no-prior \
    --tag noprior_neutral

echo "done at $(date)"
