#!/bin/bash
#PBS -N fit_diffusion_clsm
#PBS -l nodes=1:ppn=4
#PBS -l walltime=48:00:00
#PBS -q default
#PBS -j oe
#PBS -o fit_diffusion_${COND}_${PBS_JOBID}.log
#PBS -m ae
#PBS -M nishioka@ikm.uni-hannover.de

# Spatial-PDE diffusion fit (D_i, u) from CLSM/FISH z-profiles (Heine 2025 HOBIC).
# Heavy: L-BFGS minimises MSE between PDE depth-profiles and FISH data, each step
# integrating the PDE in time → dispatch to the cluster, not the workstation.
#
# Usage (one job per condition):
#   for c in CH DH; do qsub -v COND=$c fit_diffusion_clsm_job.sh; done

cd /home/nishioka/IKM_Hiwi/nife

export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORM_NAME=cpu
export JAX_ENABLE_X64=1

echo "Starting diffusion fit COND=${COND} on $(hostname) at $(date)"

/home/nishioka/IKM_Hiwi/.venv_jax/bin/python scripts/pde/fit_diffusion_clsm.py \
    --cond ${COND} \
    --data results/diffusion_fit/zprofiles_all_ti.csv \
    --Nz 40 --dt 0.05 --restarts 3

echo "Finished COND=${COND} at $(date)"
