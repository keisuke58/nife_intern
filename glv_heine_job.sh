#!/bin/bash
#PBS -N glv_heine
#PBS -l nodes=frontale02:ppn=4
#PBS -l walltime=0:30:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/glv_heine.log

export PATH="/home/nishioka/.pyenv/versions/miniconda3-latest/bin:$PATH"
cd /home/nishioka/IKM_Hiwi/nife
PYTHONUNBUFFERED=1 python3 fit_glv_heine.py
