#!/bin/bash
#SBATCH --job-name compare_order_combined
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o compare_order_combined.out
#SBATCH -e compare_order_combined.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

# Two-way feature comparison holding the order-embedding geometry fixed:
#   Order NN (ESM-C)  vs  Order NN (ESM-C + PseKRAAC)
# Both arms use the OrderEmbeddingLayer + GO-DAG order-violation loss (the
# "one true path" constraint); the ONLY thing that changes is the input
# feature vector. This tests whether classical PseKRAAC descriptors still add
# information once the labels are tied into the GO DAG by an order embedding.
#
# Override any parameter inline, e.g.:
#   LEVEL=8 SEED=7 VARIANT=hard sbatch run_compare_order_combined.sh
# For levels deeper than 4, switch -p to rtx and bump -t (~1hr per 50K proteins).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVEL=${LEVEL:-4}
SEED=${SEED:-42}
# Order-violation loss variant: "soft" (default, softplus — keeps gradients
# flowing even when the child already dominates the parent) or "hard" (ReLU).
VARIANT=${VARIANT:-soft}
# Classical feature-vector algorithms concatenated with ESM for the combined arm.
FEATURE_ALGORITHMS=${FEATURE_ALGORITHMS:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}
OUTDIR=${OUTDIR:-${WORK}/ProtCast_results/order-esm-vs-combined-level-${LEVEL}-seed-${SEED}-${VARIANT}order}

# Only use local modules for Python 3.11 to match the Python version in the container
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

echo "============================================"
echo "Order ESM vs ESM+PseKRAAC: ${VARIANT} order, level ${LEVEL}, seed ${SEED}"
echo "Features: ${FEATURE_ALGORITHMS}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/compare_order_esm_vs_order_combined.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/$EMBEDDIR-${LEVEL} \
-o $OUTDIR \
--seed $SEED \
--order-variant $VARIANT \
--feature_algorithms ${FEATURE_ALGORITHMS} \
--use_mlflow \
2>&1 | tee compare_order_combined_${VARIANT}_level_${LEVEL}_seed_${SEED}.log
