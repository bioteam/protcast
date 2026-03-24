#!/bin/bash
#SBATCH --job-name run_compare_combined_vs_embeddings
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_compare_combined_vs_embeddings.out
#SBATCH -e run_compare_combined_vs_embeddings.err
#SBATCH -N 2
#SBATCH -n 32

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
OUTDIR=${1:-comparison_experiment}
SEED=42

# Only use local modules for Python 3.11 to match the Python version in the container
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

for LEVEL in 5 6 7 8; do
    echo "============================================"
    echo "Running comparison for GO level ${LEVEL}"
    echo "============================================"
    singularity exec --nv $CONTAINER \
    python3 scripts/compare_combined_vs_embeddings.py \
    -v \
    -p $DATADIR/ProtCastDataset.bin \
    -d $DATADIR/$EMBEDDIR-${LEVEL} \
    -o $OUTDIR \
    --feature_algorithms CTriad Moran CTDD \
    --seed $SEED \
    2>&1 | tee compare_level_${LEVEL}.log
done
