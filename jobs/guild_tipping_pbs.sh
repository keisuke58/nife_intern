#!/bin/bash
#PBS -N guild_tipping_2d
#PBS -l nodes=frontale03:ppn=24
#PBS -l walltime=168:00:00
#PBS -o /home/nishioka/IKM_Hiwi/nife/logs/guild_tipping_2d.out
#PBS -e /home/nishioka/IKM_Hiwi/nife/logs/guild_tipping_2d.err

mkdir -p /home/nishioka/IKM_Hiwi/nife/logs

cd /home/nishioka/IKM_Hiwi/nife
/home/nishioka/IKM_Hiwi/.venv_jax/bin/python scripts/analysis/guild_tipping_point.py
