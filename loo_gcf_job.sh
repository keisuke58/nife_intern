#!/bin/bash
#PBS -N loo_gcf
#PBS -l nodes=1:ppn=4
#PBS -l walltime=168:00:00
#PBS -q default
#PBS -j oe
#PBS -o loo_gcf_fold${FOLD}_${PBS_JOBID}.fifawc.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# Hamilton + GCF medium (peri-implantitis enriched) LOO-CV
# GCF medium: haem×10, Gly/Pro×4-5 (collagen), Fe×6, low O2
#
# Usage:
#   for fold in 0 1 2 3 4 5 6 7 8 9; do
#     qsub -v FOLD=$fold loo_gcf_job.sh
#   done

cd /home/nishioka/IKM_Hiwi/nife
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu

echo "Hamilton GCF-medium LOO fold=${FOLD} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python scripts/loo_cv/run_hamilton_expanded_loo.py \
    --hold ${FOLD} \
    --gpu 0 \
    --use-agora \
    --agora-medium gcf \
    --fit-file fit_glv_hamilton_kegg_expanded_agora_w1p0.json \
    --tag gcf

echo "Finished fold ${FOLD} at $(date)"
