#!/bin/bash
#SBATCH --job-name run_scan_feature_vectors_only
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_scan_feature_vectors_only.out
#SBATCH -e run_scan_feature_vectors_only.err
#SBATCH -N 2
#SBATCH -n 32

# This script runs each feature vector plus the ESM-C embedding vector for GO level 8. Adjust the LEVEL variable to run for different GO levels.
# Also runs the ESM-C embedding vector alone as a baseline to compare against the combined feature vector results.

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
OUTDIR=${WORK}/ProtCast/fv_only_scan
LEVEL=8
SEED=42

# Only use local modules for Python 3.11 to match the Python version in the container
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

echo "============================================"
echo "Running FV-only scan for GO level ${LEVEL}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/scan_feature_vectors_only.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/$EMBEDDIR-${LEVEL} \
-o $OUTDIR \
--seed $SEED \
2>&1 | tee fv_only_scan_level_${LEVEL}.log
