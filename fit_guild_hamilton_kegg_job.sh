#!/bin/bash
#PBS -N hamilton_kegg_guild
#PBS -l nodes=frontale03:ppn=12
#PBS -l walltime=4:00:00
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/results/dieckow_cr/hamilton_kegg_guild.log

export PATH="/home/nishioka/IKM_Hiwi/.venv_jax/bin:$PATH"
cd /home/nishioka/IKM_Hiwi/nife

NSTEPS=${NSTEPS:-2500}
EPOCHS=${MAXITER:-5000}
LR=${LR:-3e-3}

echo "[$(date)] hamilton_kegg_guild: nsteps=${NSTEPS} epochs=${EPOCHS} lr=${LR}"

PYTHONUNBUFFERED=1 python3 loo_cv_kegg_prior.py \
    --model hamilton \
    --nsteps "${NSTEPS}" \
    --epochs "${EPOCHS}" \
    --lr     "${LR}"

echo "[$(date)] Done."
