#!/bin/bash
#SBATCH --job-name run_scan_individual_pooling
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -N 1
#SBATCH -p rtx
#SBATCH -n 16
# Full individual feature scan (~55 algorithms, --algorithms omitted = ALL).
# Submit one job per (POOLING, LEVEL); SEED via --export. Standardization is
# applied inside the classifier (scale_features=True) after the train fix.
#   sbatch -t 18:00:00 --export=ALL,POOLING=mean,LEVEL=5,SEED=42 \
#          -o run_scan_ind-mean-L5-seed-42.out -e run_scan_ind-mean-L5-seed-42.err \
#          $HOME/sh/run_scan_individual_pooling.sh

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd ${HOME}/git/ProtCast/

if [ "$POOLING" = "mean_max_std" ]; then SUF="-mean_max_std"; else SUF=""; fi
DDIR=$DATADIR/$EMBEDDIR-${LEVEL}${SUF}
OUTDIR=${WORK}/ProtCast/feature_scan_individual_standardized-${POOLING}-level-${LEVEL}-seed-${SEED}

echo "============================================"
echo "Full individual scan: pooling=${POOLING} level=${LEVEL} seed=${SEED}"
echo "data=${DDIR}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/scan_individual_features.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DDIR \
-o $OUTDIR \
--seed $SEED
