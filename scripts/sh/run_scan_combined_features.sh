#!/bin/bash
#SBATCH --job-name run_scan_combined_features
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_scan_combined_features-mean_max_std.out
#SBATCH -e run_scan_combined_features-mean-max-std.err
#SBATCH -N 2
#SBATCH -n 32

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
SEED=22

# Only use local modules for Python 3.11 to match the Python version in the container
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

for LEVEL in 5 6 7 8; do
    OUTDIR=${WORK}/ProtCast/feature_scan_combined-level-${LEVEL}-seed-${SEED}-mean_max_std

    echo "============================================"
    echo "Running combined feature scan for GO level ${LEVEL}"
    echo "============================================"
    singularity exec --nv $CONTAINER \
    python3 scripts/scan_individual_features.py \
    -v \
    -p $DATADIR/ProtCastDataset.bin \
    -d $DATADIR/$EMBEDDIR-${LEVEL}-mean_max_std \
    -o $OUTDIR \
    --seed $SEED \
    --algorithms \
        PseKRAAC_type_3B \
        PseKRAAC_type_8 \
        PseKRAAC_type_7 \
        PseKRAAC_type_3B+PseKRAAC_type_8 \
        PseKRAAC_type_3B+PseKRAAC_type_7 \
        PseKRAAC_type_8+PseKRAAC_type_7 \
        PseKRAAC_type_3B+PseKRAAC_type_8+PseKRAAC_type_7
done
