#!/bin/bash
#PBS -N hamilton_kegg
#PBS -l nodes=1:ppn=4
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dieckow_cr/hamilton_kegg.log
cd /home/nishioka/IKM_Hiwi/nife
python loo_cv_kegg_prior.py --model hamilton 2>&1
