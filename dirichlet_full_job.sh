#!/bin/bash
#PBS -N dirichlet_full
#PBS -l nodes=frontale02:ppn=4
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dirichlet_full.log

export PATH="/home/nishioka/miniconda3/bin:$PATH"
cd /home/nishioka/IKM_Hiwi/nife

PYTHONUNBUFFERED=1 /home/nishioka/IKM_Hiwi/.venv_jax/bin/python \
    fit_dirichlet_glv_pymc.py \
    --draws 1000 --tune 1000 --chains 2
