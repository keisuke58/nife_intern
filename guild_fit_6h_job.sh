#!/bin/bash
#PBS -N guild_fit_6h
#PBS -l nodes=frontale02:ppn=4
#PBS -l walltime=6:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/guild_fit_6h.log

export PATH="/home/nishioka/.pyenv/versions/miniconda3-latest/bin:$PATH"
cd /home/nishioka/IKM_Hiwi/nife
PYTHONUNBUFFERED=1 python3 fit_guild_replicator.py
