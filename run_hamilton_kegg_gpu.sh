#!/bin/bash
# Run hamilton_kegg_guild on vancouver01 GPU via SSH
# Usage: bash run_hamilton_kegg_gpu.sh [nsteps] [epochs] [lr]

NSTEPS=${1:-500}
EPOCHS=${2:-5000}
LR=${3:-3e-3}
LOGFILE=/home/nishioka/nife/results/dieckow_cr/hamilton_kegg_guild_gpu.log

echo "[$(date)] Launching hamilton_kegg_guild on vancouver01 GPU"
echo "  nsteps=${NSTEPS} epochs=${EPOCHS} lr=${LR}"
echo "  log: ${LOGFILE}"

ssh vancouver01 "
  cd /home/nishioka/nife
  source /home/nishioka/miniconda3/etc/profile.d/conda.sh
  conda activate klempt_fem
  export CUDA_VISIBLE_DEVICES=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
  nohup python3 loo_cv_kegg_prior.py \
      --model hamilton \
      --nsteps ${NSTEPS} \
      --epochs ${EPOCHS} \
      --lr     ${LR} \
      > ${LOGFILE} 2>&1 &
  echo \"PID: \$!\"
  sleep 2
  tail -5 ${LOGFILE}
"
echo "[$(date)] Job submitted. Monitor: ssh vancouver01 'tail -f /home/nishioka/nife/results/dieckow_cr/hamilton_kegg_guild_gpu.log'"
