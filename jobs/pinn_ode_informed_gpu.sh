#!/bin/bash
# 1D PINN + ODE mean-field anchor on vancouver01 GPU.
# Infers D_i (5-species) and u from Heine FISH depth profiles,
# using the gLV ODE continuous trajectory as temporal anchor.
#
# Usage:
#   bash jobs/pinn_ode_informed_gpu.sh CH
#   bash jobs/pinn_ode_informed_gpu.sh DH   (default)
#   bash jobs/pinn_ode_informed_gpu.sh both  (CH then DH sequentially)

COND=${1:-DH}
EPOCHS=6000
HIDDEN=64
DEPTH=4
COLLOCATION=1024
W_DATA=1.0
W_PHYS=1.0
W_ODE=5.0
N_Z_QUAD=32
N_ODE_PER_EPOCH=128   # 421 pts total; 128/epoch = 30% temporal coverage per epoch
LR=2e-3
SEED=0

LOGDIR=/home/nishioka/nife/results/pinn_ode_informed
ODE_DIR=/home/nishioka/nife/results/heine2025

run_cond() {
    local cond=$1
    local LOGFILE="${LOGDIR}/pinn_ode_${cond}_gpu.log"
    echo "[$(date)] Launching PINN+ODE --cond ${cond} on vancouver01 GPU"

    ssh vancouver01 "
      mkdir -p ${LOGDIR}
      cd /home/nishioka/nife
      source /home/nishioka/miniconda3/etc/profile.d/conda.sh
      conda activate klempt_fem
      export CUDA_VISIBLE_DEVICES=0
      export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
      export JAX_ENABLE_X64=1
      nohup python3 scripts/pde/pinn_1d_ode_informed.py \
          --cond              ${cond} \
          --ode-dir           ${ODE_DIR} \
          --epochs            ${EPOCHS} \
          --hidden            ${HIDDEN} \
          --depth             ${DEPTH} \
          --collocation       ${COLLOCATION} \
          --w-data            ${W_DATA} \
          --w-phys            ${W_PHYS} \
          --w-ode             ${W_ODE} \
          --n-z-quad          ${N_Z_QUAD} \
          --n-ode-per-epoch   ${N_ODE_PER_EPOCH} \
          --lr                ${LR} \
          --seed              ${SEED} \
          > ${LOGFILE} 2>&1 &
      echo \"PID: \$!\"
      sleep 3
      tail -8 ${LOGFILE}
    "
    echo "[$(date)] ${cond} submitted. Monitor:"
    echo "  ssh vancouver01 'tail -f ${LOGFILE}'"
}

if [ "${COND}" = "both" ]; then
    run_cond CH
    echo ""
    echo "Waiting 10 s before DH ..."
    sleep 10
    run_cond DH
else
    run_cond "${COND}"
fi
