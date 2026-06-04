#!/bin/bash
# Submit neutral-init no-prior LOO: all 10 folds
# Usage: bash jobs/loo_noprior_neutral_submit.sh
set -e
cd "$(dirname "$0")/.."
for fold in $(seq 0 9); do
    qsub -v FOLD=$fold jobs/loo_noprior_neutral_job.sh
    echo "submitted fold $fold"
done
