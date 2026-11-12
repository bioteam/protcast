#!/bin/bash
#SBATCH --job-name run_scan_candidates
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -N 1
#SBATCH -p rtx
#SBATCH -n 16
# Multi-seed significance test of candidate feature vectors (those that beat the
# standardized baseline by >=0.004 in the single-seed scan). Submit per (POOLING, SEED).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd ${HOME}/git/ProtCast/

if [ "$POOLING" = "mean_max_std" ]; then SUF="-mean_max_std"; else SUF=""; fi
for LEVEL in 5 6 7 8; do
  DDIR=$DATADIR/$EMBEDDIR-${LEVEL}${SUF}
  OUTDIR=${WORK}/ProtCast/feature_scan_candidates-${POOLING}-level-${LEVEL}-seed-${SEED}
  echo "=== candidates: pooling=${POOLING} level=${LEVEL} seed=${SEED} ==="
  singularity exec --nv $CONTAINER \
  python3 scripts/scan_individual_features.py -v \
  -p $DATADIR/ProtCastDataset.bin \
  -d $DDIR -o $OUTDIR --seed $SEED \
  --algorithms GAAC PseKRAAC_type_13 PseKRAAC_type_9 PseKRAAC_type_1 PseKRAAC_type_11 PseKRAAC_type_14 PseKRAAC_type_6C PseKRAAC_type_16 PseKRAAC_type_10 PseKRAAC_type_6B
done
