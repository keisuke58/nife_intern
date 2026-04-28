#!/bin/bash
#PBS -N heine_kegg_prior
#PBS -l nodes=1:ppn=4
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/heine2025/heine_kegg_prior.log
cd /home/nishioka/IKM_Hiwi/nife
/home/nishioka/IKM_Hiwi/.venv_jax/bin/python fit_glv_heine_kegg_prior.py 2>&1
