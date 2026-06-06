#!/bin/bash
# 1D PINN inverse problem on vancouver01 GPU — saves network weights (.npz)
# alongside the D_fit JSON so fig_pde_pinn_fit.py can use the true forward pass.
#
# Usage:
#   bash jobs/pinn_1d_gpu.sh CH
#   bash jobs/pinn_1d_gpu.sh DH
#   bash jobs/pinn_1d_gpu.sh   # defaults to both

CONDS=${@:-"CH DH"}
EPOCHS=5000
HIDDEN=64
DEPTH=4
COLLOCATION=512
W_DATA=1.0
W_PHYS=50.0
LR=2e-3

LOGDIR=/home/nishioka/nife/results/pinn_diffusion

for COND in ${CONDS}; do
  LOGFILE="${LOGDIR}/pinn1d_${COND}_prod.log"
  echo "[$(date)] Launching 1D PINN --cond ${COND} on vancouver01 GPU"

  ssh vancouver01 "
    mkdir -p ${LOGDIR}
    cd /home/nishioka/nife
    source /home/nishioka/miniconda3/etc/profile.d/conda.sh
    conda activate klempt_fem
    export CUDA_VISIBLE_DEVICES=0
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
    export JAX_ENABLE_X64=1
    nohup python3 scripts/pde/pinn_diffusion_inverse.py \
        --cond        ${COND} \
        --epochs      ${EPOCHS} \
        --hidden      ${HIDDEN} \
        --depth       ${DEPTH} \
        --collocation ${COLLOCATION} \
        --w-data      ${W_DATA} \
        --w-phys      ${W_PHYS} \
        --lr          ${LR} \
        > ${LOGFILE} 2>&1 &
    echo \"PID: \$!\"
    sleep 3
    tail -8 ${LOGFILE}
  "
  echo "[$(date)] ${COND} submitted. Monitor:"
  echo "  ssh vancouver01 'tail -f ${LOGFILE}'"
done
