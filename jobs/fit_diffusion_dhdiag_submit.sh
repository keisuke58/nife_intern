#!/bin/bash
# fit_diffusion_dhdiag_submit.sh — DIAGNOSE why the DH diffusion fit will not
# converge to a clean low-loss solution. Diagnosis so far (2026-06-04):
#   * DH loss floors at ~0.09 for BOTH success=True (fast_nz24=0.091) and
#     success=False (nz32=0.0996) runs -> the floor is INTRINSIC (the
#     reaction-dominated PDE cannot reproduce DH depth stratification), not an
#     optimiser bug; success=False is mostly a noisy-numerical-gradient flag.
#   * The best CH run pins D[An] at the upper bound 0.5 -> weak identifiability.
# This sweep (DH only) separates the two: the fast recipe (relaxed ftol, many
# restarts) should give success=True AT the floor, and varying d_max (0.5 vs 1.0)
# tests boundary pinning. Each job ~2h; tagged so nothing collides with the
# running sweep. Dispatch + retrieve only (shared-cluster courtesy).
#
#   bash jobs/fit_diffusion_dhdiag_submit.sh
set -eu
cd /home/nishioka/IKM_Hiwi/nife

# fast, reliable-convergence recipe: small Nz, large dt, relaxed ftol, big fd_eps,
# many restarts (16) to escape boundary basins and reach the loss floor.
DT=0.10; RESTARTS=16; MAXITER=250; FDEPS=2e-3; FTOL=1e-8

for NZ in 24 32; do
  for DMAX in 0.5 1.0; do
    DTAG=$(echo "d${DMAX}" | tr '.' 'p')          # d0p5 / d1p0
    TAG="dhdiag_nz${NZ}_${DTAG}"
    qsub -v COND=DH,NZ=$NZ,DT=$DT,RESTARTS=$RESTARTS,MAXITER=$MAXITER,FDEPS=$FDEPS,FTOL=$FTOL,DMAX=$DMAX,TAG=$TAG \
         jobs/fit_diffusion_sweep_job.sh
    echo "submitted: DH Nz=$NZ d_max=$DMAX restarts=$RESTARTS fd_eps=$FDEPS ftol=$FTOL tag=$TAG"
  done
done
