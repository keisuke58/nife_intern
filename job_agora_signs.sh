#!/bin/bash
#PBS -N agora_signs
#PBS -l nodes=1:ppn=4
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dieckow_cr/agora_signs.log
cd /home/nishioka/IKM_Hiwi/nife

# Step 1: download models if not present
/home/nishioka/miniconda3/bin/python download_agora2_oral.py \
    --out_dir data/agora2_xml 2>&1

# Step 2: run sign validation
/home/nishioka/miniconda3/bin/python guild_agora_signs.py \
    --agora_dir data/agora2_xml \
    --glv_fit results/dieckow_cr/fit_glv_8pat_kegg_prior.json \
    --out_dir /home/nishioka/IKM_Hiwi/docs/figures/dieckow \
    --plot 2>&1
