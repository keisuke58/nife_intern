#!/bin/bash
#PBS -N loo_micom_perf
#PBS -l nodes=1:ppn=4
#PBS -l walltime=168:00:00
#PBS -q default
#PBS -j oe
#PBS -o loo_micom_perf_fold${FOLD}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# LOO-CV with MICOM "perfect" version:
#   - fraction=0.5  (Diener 2020 default, was 0.3)
#   - flux magnitude weighting  (was binary count)
#   - toxins harm all co-occurring guilds  (was consumer-only)
#
# Usage:
#   for i in 0 1 2 3 4 5 6 7 8 9; do
#     qsub -v FOLD=$i loo_micom_perfect_job.sh
#   done

cd /home/nishioka/IKM_Hiwi/nife

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=4"

echo "Starting MICOM-perfect LOO fold ${FOLD} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python scripts/loo_cv/run_hamilton_expanded_loo.py \
    --hold ${FOLD} \
    --gpu -1 \
    --fit-file fit_glv_hamilton_kegg_expanded_agora_w1p0.json \
    --use-agora \
    --agora-medium micom \
    --micom-fraction 0.5 \
    --tag micom_perf

echo "Finished fold ${FOLD} at $(date)"
