#!/bin/bash
#SBATCH --job-name run_scan_individual_features
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_scan_individual_features.out
#SBATCH -e run_scan_individual_features.err
#SBATCH -N 2
#SBATCH -n 32

# This script runs the feature scan for individual features (not combined) for GO level 5. Adjust the LEVEL variable to run for different GO levels.


CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVEL=5
SEED=42
OUTDIR=${WORK}/ProtCast/feature_scan-level-${LEVEL}-seed-${SEED}

# Only use local modules for Python 3.11 to match the Python version in the container
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

echo "============================================"
echo "Running feature scan for GO level ${LEVEL}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/scan_individual_features.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/$EMBEDDIR-${LEVEL} \
-o $OUTDIR \
--seed $SEED
