#!/bin/bash
# Dieckow 10-patient posterior bundle on vancouver01 GPU.
# Produces continuous-time trajectories + CI bands for all 10 patients.
#
# Usage:
#   bash jobs/dieckow_ode_continuous_gpu.sh          # default 200 samples
#   bash jobs/dieckow_ode_continuous_gpu.sh --full   # all 1000 samples (~2-3 min)

EXTRA_ARGS="$@"
LOGFILE=/home/nishioka/nife/results/dieckow_ode_continuous/gpu_run.log

echo "[$(date)] Launching dieckow_ode_continuous on vancouver01 GPU"

ssh vancouver01 "
  mkdir -p /home/nishioka/nife/results/dieckow_ode_continuous
  cd /home/nishioka/nife
  source /home/nishioka/miniconda3/etc/profile.d/conda.sh
  conda activate klempt_fem
  export CUDA_VISIBLE_DEVICES=0
  export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
  export JAX_ENABLE_X64=1
  nohup python3 scripts/analysis/dieckow_ode_continuous.py ${EXTRA_ARGS} \
      > ${LOGFILE} 2>&1 &
  echo \"PID: \$!\"
  sleep 5
  tail -10 ${LOGFILE}
"
echo "[$(date)] Submitted. Monitor:"
echo "  ssh vancouver01 'tail -f ${LOGFILE}'"
