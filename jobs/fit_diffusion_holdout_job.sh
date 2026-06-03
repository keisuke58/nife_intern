#!/bin/bash
#PBS -N fit_diff_holdout
#PBS -l nodes=1:ppn=4
#PBS -l walltime=48:00:00
#PBS -q default
#PBS -j oe
#PBS -o fit_diff_holdout_${COND}_d${HOLDOUT}_${PBS_JOBID}.log
#PBS -m n

# Spatial leave-one-DAY-out validation for the diffusion fit (Heine 2025 HOBIC).
# Fit D_i,u on all FISH days except the held-out day, then PREDICT the held-out
# day's depth profile and report the held-out RMSE (spatial analogue of LOO-CV).
# Heavy: each L-BFGS step integrates the reaction-diffusion PDE -> dispatch to HPC.
#
# Parameters via -v, with defaults.
# Usage:
#   for c in CH DH; do qsub -v COND=$c,HOLDOUT=21 fit_diffusion_holdout_job.sh; done

cd /home/nishioka/IKM_Hiwi/nife

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=1

COND=${COND:-DH}; HOLDOUT=${HOLDOUT:-21}
NZ=${NZ:-40}; DT=${DT:-0.05}; RESTARTS=${RESTARTS:-8}
FDEPS=${FDEPS:-1e-3}; MAXITER=${MAXITER:-300}; FTOL=${FTOL:-1e-9}
XDIFF=""; [ "${CROSSDIFF:-0}" = "1" ] && XDIFF="--crossdiff"

echo "Holdout COND=${COND} day=${HOLDOUT} Nz=${NZ} dt=${DT} restarts=${RESTARTS} fd_eps=${FDEPS} crossdiff=${CROSSDIFF:-0} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python scripts/pde/fit_diffusion_holdout.py \
    --cond ${COND} \
    --holdout-day ${HOLDOUT} \
    --data results/diffusion_fit/zprofiles_all_ti.csv \
    --Nz ${NZ} --dt ${DT} --restarts ${RESTARTS} \
    --maxiter ${MAXITER} --ftol ${FTOL} --gtol 1e-6 --fd-eps ${FDEPS} \
    ${XDIFF}

echo "Finished COND=${COND} day=${HOLDOUT} at $(date)"
