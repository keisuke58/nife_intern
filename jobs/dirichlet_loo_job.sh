#!/bin/bash
#PBS -N dirichlet_loo
#PBS -l nodes=marinos01:ppn=4
#PBS -l walltime=12:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dirichlet_loo.log

export PATH="/home/nishioka/miniconda3/bin:$PATH"
cd /home/nishioka/IKM_Hiwi/nife

mkdir -p results/dirichlet_loo

PYTHONUNBUFFERED=1 /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
    scripts/loo_cv/loo_cv_dirichlet_pymc.py --all
