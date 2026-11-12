#!/bin/bash
#SBATCH --job-name run_scan_pooling
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -N 1
#SBATCH -p rtx
#SBATCH -t 06:00:00
#SBATCH -n 16
# Re-run of the combined PseKRAAC scan with the train-fold StandardScaler fix.
# Submit one job per (POOLING, SEED); each loops GO levels 5-8.
#   sbatch --export=ALL,POOLING=mean,SEED=1 \
#          -o run_scan_pooling-mean-seed-1.out -e run_scan_pooling-mean-seed-1.err \
#          $HOME/sh/run_scan_pooling.sh
# POOLING is "mean" (1152-d) or "mean_max_std" (3456-d).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd ${HOME}/git/ProtCast/

if [ "$POOLING" = "mean_max_std" ]; then SUF="-mean_max_std"; else SUF=""; fi

for LEVEL in 5 6 7 8; do
    DDIR=$DATADIR/$EMBEDDIR-${LEVEL}${SUF}
    OUTDIR=${WORK}/ProtCast/feature_scan_standardized-${POOLING}-level-${LEVEL}-seed-${SEED}
    echo "============================================"
    echo "Rescaled combined scan: pooling=${POOLING} level=${LEVEL} seed=${SEED}"
    echo "data=${DDIR}"
    echo "============================================"
    singularity exec --nv $CONTAINER \
    python3 scripts/scan_individual_features.py \
    -v \
    -p $DATADIR/ProtCastDataset.bin \
    -d $DDIR \
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
