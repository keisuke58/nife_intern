#!/bin/bash
#PBS -N compare_signs
#PBS -l nodes=1:ppn=2
#PBS -l walltime=02:00:00
#PBS -q default
#PBS -j oe
#PBS -o compare_signs_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

cd /home/nishioka/IKM_Hiwi/nife
export CUDA_VISIBLE_DEVICES=""
echo "compare_agora_v1_v2 (with MICOM perfect) on $(hostname) at $(date)"
/home/nishioka/IKM_Hiwi/.venv_jax/bin/python compare_agora_v1_v2.py
echo "Done at $(date)"
