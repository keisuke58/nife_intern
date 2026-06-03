#!/bin/bash
#PBS -N guild_fit
#PBS -l nodes=frontale01:ppn=4
#PBS -l walltime=2:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/guild_fit.log

export PATH="/home/nishioka/.pyenv/versions/miniconda3-latest/bin:$PATH"
cd /home/nishioka/IKM_Hiwi/nife
PYTHONUNBUFFERED=1 python3 scripts/fitting/fit_guild_replicator.py
