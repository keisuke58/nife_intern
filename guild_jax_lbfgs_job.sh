#!/bin/bash
#PBS -N guild_jax_lbfgs
#PBS -l nodes=frontale01:ppn=4
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/guild_jax_lbfgs.log

export PATH="/home/nishioka/.pyenv/versions/miniconda3-latest/bin:$PATH"
cd /home/nishioka/IKM_Hiwi/nife
PYTHONUNBUFFERED=1 python3 fit_guild_jax_lbfgs.py
