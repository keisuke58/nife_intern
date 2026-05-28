#!/bin/bash
#PBS -N guild_network_perm
#PBS -l nodes=frontale01:ppn=24
#PBS -l walltime=168:00:00
#PBS -o /home/nishioka/IKM_Hiwi/nife/logs/guild_network_perm.out
#PBS -e /home/nishioka/IKM_Hiwi/nife/logs/guild_network_perm.err

mkdir -p /home/nishioka/IKM_Hiwi/nife/logs

cd /home/nishioka/IKM_Hiwi/nife
/home/nishioka/IKM_Hiwi/.venv_jax/bin/python guild_network_analysis.py --n-perm 100
