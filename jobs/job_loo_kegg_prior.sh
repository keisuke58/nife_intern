#!/bin/bash
#PBS -N loo_kegg_prior
#PBS -l nodes=1:ppn=4
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dieckow_cr/loo_kegg_prior.log
cd /home/nishioka/IKM_Hiwi/nife
python loo_cv_kegg_prior.py --model glv 2>&1
