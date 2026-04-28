#!/bin/bash
#PBS -N hamilton_kegg_fixed
#PBS -l nodes=1:ppn=4
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dieckow_cr/hamilton_kegg_fixed.log
cd /home/nishioka/IKM_Hiwi/nife
/home/nishioka/IKM_Hiwi/.venv_jax/bin/python loo_cv_kegg_prior.py --model hamilton 2>&1
