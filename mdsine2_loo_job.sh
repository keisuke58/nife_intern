#!/bin/bash
#PBS -N mdsine2_loo
#PBS -l nodes=1:ppn=4
#PBS -l walltime=168:00:00
#PBS -q default
#PBS -j oe
#PBS -o mdsine2_loo_fold${FOLD}_${PBS_JOBID}.log

cd /home/nishioka/IKM_Hiwi/nife
export CUDA_VISIBLE_DEVICES=""
export NUMBA_DISABLE_CACHING=1
export NUMBA_CACHE_DIR=/tmp/numba_cache_$$

echo "MDSINE2 LOO fold ${FOLD} on $(hostname) at $(date)"
/home/nishioka/IKM_Hiwi/.venv_jax/bin/python run_mdsine2_loo.py \
    --fold ${FOLD} --burnin 500 --n_samples 2500
echo "Done at $(date)"
