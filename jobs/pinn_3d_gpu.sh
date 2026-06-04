#!/bin/bash
# 3D PINN inverse problem on vancouver01 GPU — production run.
# Infers D_i (5-species) and u from Heine 2025 HOBIC FISH 3-D voxel stacks.
#
# Usage:
#   bash jobs/pinn_3d_gpu.sh CH   # commensal
#   bash jobs/pinn_3d_gpu.sh DH   # dysbiotic (default)

COND=${1:-DH}
EPOCHS=3000
HIDDEN=64
DEPTH=4
N_DATA=300000
BATCH_DATA=4096
COLLOCATION=1024
W_DATA=1.0
W_PHYS=5.0
LR=2e-3

LOGDIR=/home/nishioka/nife/results/pinn_3d
LOGFILE="${LOGDIR}/pinn3d_${COND}_prod.log"

echo "[$(date)] Launching 3D PINN --cond ${COND} on vancouver01 GPU"

ssh vancouver01 "
  mkdir -p ${LOGDIR}
  cd /home/nishioka/nife
  source /home/nishioka/miniconda3/etc/profile.d/conda.sh
  conda activate klempt_fem
  export CUDA_VISIBLE_DEVICES=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6
  export JAX_ENABLE_X64=1
  nohup python3 scripts/pde/pinn_3d_inverse.py \
      --cond        ${COND} \
      --epochs      ${EPOCHS} \
      --hidden      ${HIDDEN} \
      --depth       ${DEPTH} \
      --n-data      ${N_DATA} \
      --batch-data  ${BATCH_DATA} \
      --collocation ${COLLOCATION} \
      --w-data      ${W_DATA} \
      --w-phys      ${W_PHYS} \
      --lr          ${LR} \
      > ${LOGFILE} 2>&1 &
  echo \"PID: \$!\"
  sleep 3
  tail -8 ${LOGFILE}
"
echo "[$(date)] Job submitted. Monitor:"
echo "  ssh vancouver01 'tail -f ${LOGFILE}'"
