#!/bin/bash
#PBS -N comets_BC
#PBS -q default
#PBS -l nodes=frontale02:ppn=4
#PBS -o /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/comets_BC.log
#PBS -e /home/nishioka/IKM_Hiwi/nife/comets/pipeline_results/comets_BC.err

set -eo pipefail

source /home/nishioka/miniconda3/etc/profile.d/conda.sh || true
conda activate base || true
set -u

# cometspy lives in .venv_jax
source /home/nishioka/IKM_Hiwi/.venv_jax/bin/activate

cd /home/nishioka/IKM_Hiwi
export COMETS_HOME=/home/nishioka/comets_linux
OUTDIR=/home/nishioka/IKM_Hiwi/nife/comets/pipeline_results

echo "[$(date)] === Step B: 2D spatial COMETS ==="
python3 nife/comets/run_comets_pipeline.py \
    --step B \
    --cycles 800 \
    --nx 10 \
    --nz 20 \
    --outdir $OUTDIR

echo "[$(date)] === Step C: MetaPhlAn init_comp ==="
INIT_COMP=$OUTDIR/../../../data/metaphlan_profiles/init_comp_ERR13166576_A_3.json
python3 nife/comets/run_comets_pipeline.py \
    --step C \
    --cycles 500 \
    --init-comp $INIT_COMP \
    --outdir $OUTDIR

echo "[$(date)] === B + C done ==="
echo "Results: $OUTDIR"
ls -lh $OUTDIR/*.png
