#!/bin/bash
# Submit the diffusion-fit hyperparameter sweep to PBS.
# Grid: condition × finite-diff step (fd_eps, the convergence-critical knob) ×
# spatial resolution Nz. The Nz=40 / fd_eps=1e-3 baseline already runs as the
# canonical (untagged) job, so it is omitted here.
set -e
cd "$(dirname "$0")"

for COND in CH DH; do
  for NZ in 32 48; do
    for FDEPS in 3e-3 1e-3 3e-4; do
      TAG="nz${NZ}_eps${FDEPS}"
      echo "qsub COND=$COND $TAG"
      qsub -v COND=$COND,NZ=$NZ,FDEPS=$FDEPS,TAG=$TAG fit_diffusion_sweep_job.sh
    done
  done
done
