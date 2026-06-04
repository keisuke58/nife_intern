#!/bin/bash
#PBS -N fit_diff_sweep
#PBS -l nodes=1:ppn=4
#PBS -l walltime=48:00:00
#PBS -q default
#PBS -j oe
#PBS -o fit_diff_sweep_${COND}_${TAG}_${PBS_JOBID}.log
#PBS -m n

# Hyperparameter sweep for the diffusion fit. Each job writes tagged outputs
# (D_fit_<COND>_<TAG>.json / fit_<COND>_<TAG>.png) so combos do not collide.
# Parameters via -v, with defaults; TAG should encode the combo.
#
# Usage (see fit_diffusion_sweep_submit.sh for the full grid):
#   qsub -v COND=CH,NZ=48,FDEPS=3e-3,TAG=nz48_eps3e3 fit_diffusion_sweep_job.sh

cd /home/nishioka/IKM_Hiwi/nife

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=1

NZ=${NZ:-40}; DT=${DT:-0.05}; RESTARTS=${RESTARTS:-8}
FDEPS=${FDEPS:-1e-3}; MAXITER=${MAXITER:-300}; FTOL=${FTOL:-1e-9}; DMAX=${DMAX:-0.5}
XDIFF=""; [ "${CROSSDIFF:-0}" = "1" ] && XDIFF="--crossdiff"

echo "Sweep COND=${COND} TAG=${TAG} Nz=${NZ} dt=${DT} restarts=${RESTARTS} fd_eps=${FDEPS} d_max=${DMAX} crossdiff=${CROSSDIFF:-0} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python scripts/pde/fit_diffusion_clsm.py \
    --cond ${COND} \
    --data results/diffusion_fit/zprofiles_all_ti.csv \
    --Nz ${NZ} --dt ${DT} --restarts ${RESTARTS} \
    --maxiter ${MAXITER} --ftol ${FTOL} --gtol 1e-6 --fd-eps ${FDEPS} --d-max ${DMAX} \
    ${XDIFF} --tag ${TAG}

echo "Finished COND=${COND} TAG=${TAG} at $(date)"
