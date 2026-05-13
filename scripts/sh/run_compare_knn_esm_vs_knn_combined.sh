#!/bin/bash
#SBATCH --job-name run_knn_esm_vs_combined
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_knn_esm_vs_combined.out
#SBATCH -e run_knn_esm_vs_combined.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

# Two-way comparison: KNN(ESM-C) vs KNN(ESM-C + classical FVs).
# Computes classical feature vectors (CTriad, Moran, CTDD) once, then trains
# both KNNs on identical train/val splits so any Fmax delta is attributable
# to the feature representation alone. For deeper levels switch -p to rtx
# and bump -t (classical FV computation scales with sequence length).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVEL=4
SEED=42
OUTDIR=${WORK}/ProtCast_results/knn_esm_vs_combined-level-${LEVEL}-seed-${SEED}

# Only use local modules for Python 3.11 to match the Python version in the container
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

echo "============================================"
echo "Running KNN(ESM-C) vs KNN(ESM-C + classical) for GO level ${LEVEL}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/compare_knn_esm_vs_knn_combined.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/$EMBEDDIR-${LEVEL} \
-o $OUTDIR \
--feature_algorithms CTriad Moran CTDD \
--seed $SEED \
--use_mlflow \
2>&1 | tee knn_esm_vs_combined_level_${LEVEL}.log
