#!/bin/bash
#PBS -N nife_comets_java_test
#PBS -l nodes=1:ppn=1
#PBS -j oe
#PBS -o /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/comets_java_test.log

set -euo pipefail

CONDA_BASE="$(conda info --base 2>/dev/null || echo /home/nishioka/miniconda3)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate metaphlan4

export COMETS_HOME="${HOME}/comets_2.12.3"
export JAVA_HOME="$(dirname $(dirname $(which java)))"

cd /home/nishioka/IKM_Hiwi

PYTHON=/home/nishioka/IKM_Hiwi/.venv_jax/bin/python3
echo "[$(date)] Testing COMETS Java with AGORA-enriched medium ..."
"${PYTHON}" /home/nishioka/IKM_Hiwi/nife/comets/test_comets_java_inner.py

echo "[$(date)] Finished."
