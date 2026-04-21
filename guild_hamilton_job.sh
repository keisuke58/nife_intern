#!/bin/bash
#PBS -N guild_hamilton
#PBS -l nodes=frontale03:ppn=4
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/guild_hamilton.log

export PATH="/home/nishioka/.pyenv/versions/miniconda3-latest/bin:$PATH"
export JAX_PLATFORMS=cpu
cd /home/nishioka/IKM_Hiwi/nife
PYTHONUNBUFFERED=1 python3 fit_guild_hamilton.py
