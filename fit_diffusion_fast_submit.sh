#!/bin/bash
# fit_diffusion_fast_submit.sh — submit a small set of FAST (~2h) diffusion fits
# alongside the running sweep, WITHOUT touching existing jobs. Settings chosen for
# quick, reliable convergence: coarse grid (small Nz) + larger dt (fewer time
# steps) + relaxed ftol/fd_eps (numerical-gradient noise) + more restarts.
#
# Each combo writes tagged outputs (D_fit_<COND>_<TAG>.json / fit_<COND>_<TAG>.png)
# so nothing collides with the running Nz32/48 sweep or the CH/DH baselines.
#
#   bash fit_diffusion_fast_submit.sh
set -eu
cd /home/nishioka/IKM_Hiwi/nife

# combo: NZ DT RESTARTS MAXITER FDEPS FTOL TAG
combos=(
  "24 0.10 12 200 2e-3 1e-8 fast_nz24"
  "30 0.08 10 200 2e-3 1e-8 fast_nz30"
)

for cond in CH DH; do
  for c in "${combos[@]}"; do
    set -- $c
    NZ=$1; DT=$2; RESTARTS=$3; MAXITER=$4; FDEPS=$5; FTOL=$6; TAG=$7
    qsub -v COND=$cond,NZ=$NZ,DT=$DT,RESTARTS=$RESTARTS,MAXITER=$MAXITER,FDEPS=$FDEPS,FTOL=$FTOL,TAG=$TAG \
         fit_diffusion_sweep_job.sh
    echo "submitted: $cond Nz=$NZ dt=$DT restarts=$RESTARTS maxiter=$MAXITER fd_eps=$FDEPS ftol=$FTOL tag=$TAG"
  done
done
