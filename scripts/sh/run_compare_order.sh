#!/bin/bash
#SBATCH --job-name compare_order
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o compare_order.out
#SBATCH -e compare_order.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

# Three-way geometry comparison: KNN vs MultiLabel flat vs MultiLabel+order
# (order-embedding NN) on ESM-C embeddings.  The order model represents each
# GO term as a point in the non-negative orthant and ties the labels into the
# GO DAG via the reversed product order, replacing the box-embedding geometry.
#
# Override any parameter inline, e.g.:   LEVEL=8 SEED=7 VARIANT=hard sbatch run_compare_order.sh
# For levels deeper than 4, switch -p to rtx and bump -t (~1hr per 50K proteins).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVEL=${LEVEL:-4}
SEED=${SEED:-42}
# Order-violation loss variant: "soft" (default, softplus — keeps gradients
# flowing even when the child already dominates the parent) or "hard" (ReLU).
VARIANT=${VARIANT:-soft}
OUTDIR=${OUTDIR:-${WORK}/ProtCast_results/knn_vs_multilabel-level-${LEVEL}-seed-${SEED}-${VARIANT}order}

# Only use local modules for Python 3.11 to match the Python version in the container
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

echo "============================================"
echo "Geometry comparison: ${VARIANT} order, level ${LEVEL}, seed ${SEED}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/compare_knn_vs_multilabel.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/$EMBEDDIR-${LEVEL} \
-o $OUTDIR \
--seed $SEED \
--order \
--order-variant $VARIANT \
--use_mlflow \
2>&1 | tee compare_${VARIANT}order_level_${LEVEL}_seed_${SEED}.log
