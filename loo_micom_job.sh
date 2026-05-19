#!/bin/bash
#PBS -N loo_micom
#PBS -l nodes=1:ppn=4
#PBS -l walltime=168:00:00
#PBS -q default
#PBS -j oe
#PBS -o loo_micom_fold${FOLD}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# LOO-CV with AGORA MICOM (community FBA cooperative tradeoff)
# Usage:
#   for i in 0 1 2 3 4 5 6 7 8 9; do
#     qsub -v FOLD=$i loo_micom_job.sh
#   done

cd /home/nishioka/IKM_Hiwi/nife

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=4"

echo "Starting MICOM LOO fold ${FOLD} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python run_hamilton_expanded_loo.py \
    --hold ${FOLD} \
    --gpu -1 \
    --fit-file fit_glv_hamilton_kegg_expanded_agora_w1p0.json \
    --use-agora \
    --agora-medium micom \
    --tag micom

echo "Finished fold ${FOLD} at $(date)"
