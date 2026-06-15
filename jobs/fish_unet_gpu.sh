#!/bin/bash
# Full-resolution 3-D U-Net training for FISH segmentation on vancouver01 GPU.
# Companion to portfolio/fish_unet_train.py (the CPU run is downsampled).
# Usage: bash jobs/fish_unet_gpu.sh [epochs] [base] [lr]
#   epochs : training epochs           (default 200)
#   base   : U-Net base channel width  (default 32; GPU can afford it)
#   lr     : Adam learning rate        (default 1e-3)
#
# NOTE: shared GPU host — do NOT flood. Submit one run, monitor, collect.

EPOCHS=${1:-200}
BASE=${2:-32}
LR=${3:-1e-3}
LOGFILE=/home/nishioka/nife/portfolio/fish_unet_gpu.log

echo "[$(date)] Launching fish_unet_train (full-res) on vancouver01 GPU"
echo "  epochs=${EPOCHS} base=${BASE} lr=${LR}"
echo "  log: ${LOGFILE}"

ssh vancouver01 "
  cd /home/nishioka/nife
  source /home/nishioka/miniconda3/etc/profile.d/conda.sh
  conda activate gnn_final_env   # torch 2.7.1+cu126, CUDA available
  export CUDA_VISIBLE_DEVICES=0
  nohup python3 portfolio/fish_unet_train.py \
      --epochs ${EPOCHS} \
      --downsample 1 \
      --zpatch 32 \
      --base ${BASE} \
      --lr ${LR} \
      > ${LOGFILE} 2>&1 &
  echo \"PID: \$!\"
  sleep 2
  tail -5 ${LOGFILE}
"
echo "[$(date)] Job submitted. Monitor: ssh vancouver01 'tail -f ${LOGFILE}'"
