#!/bin/bash
#SBATCH --job-name compare_dual_encoder
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o compare_dual_encoder.out
#SBATCH -e compare_dual_encoder.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

# Pilot script for the dual-encoder order NN experiment.
# Runs a single (level, seed) pair via the compare_knn_vs_multilabel.py
# --dual-encoder flag, which adds a dedicated PseKRAAC MLP branch to the
# order NN architecture. This prevents the 12 PseKRAAC dimensions from being
# drowned out by the 1152 ESM-C dimensions in the first weight matrix.
#
# The run produces four models and saves results under a single JSON:
#   1. KNN (ESM-C only, cosine)
#   2. Flat NN (ESM-C only)
#   3. Order NN (ESM-C only, --order)
#   4. Order NN + dual-encoder (ESM-C + PseKRAAC, --dual-encoder)
#
# This is intentionally a pilot at level 6 / seed 42 — the level where the
# earlier single-encoder combined experiment showed the most promising
# (though still small) PseKRAAC gain. Run this first; only launch the full
# sweep if the dual-encoder model shows a meaningful Fmax improvement over
# the ESM-only order NN.
#
# Override any parameter inline, e.g.:
#   LEVEL=7 SEED=43 sbatch run_compare_dual_encoder.sh
# For a multi-level sweep use launch_dual_encoder_sweep_all_levels_per_seed.sh.

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
# Pooling suffix on the embedding directory: mf_go_terms-level-<N>-<POOL>.
# Empty POOL falls back to the legacy plain mf_go_terms-level-<N> layout.
POOL=${POOL:-mean_max_std}
LEVEL=${LEVEL:-6}
SEED=${SEED:-42}
VARIANT=${VARIANT:-soft}
FEATURE_ALGORITHMS=${FEATURE_ALGORITHMS:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}
POOL_SUFFIX=${POOL:+-${POOL}}
OUTDIR=${OUTDIR:-${WORK}/ProtCast_results/knn_vs_multilabel-${POOL:-mean}-level-${LEVEL}-seed-${SEED}-${VARIANT}order-dual}
# Scale the single-encoder baseline arms (flat/order). The dual arm already
# scales its ESM/FV blocks, so this is a no-op there but keeps the embeddings-
# only NN baseline fair for the FV-value delta. MANDATORY for mean_max_std.
SCALE_FEATURES=${SCALE_FEATURES:-1}
SCALE_ARG=""; [ "$SCALE_FEATURES" = "1" ] && SCALE_ARG="--scale-features"

# Prepend the repo root so `import protcast` resolves to THIS checkout, not a
# stale pip-installed copy in ~/.local (scripts are run by path, so the repo
# root is otherwise never on sys.path). Prevents new protcast API — e.g.
# MultiLabelClassifier(scale_features=...) — from silently hitting old code.
export PYTHONPATH=/work2/10504/wisdawg/frontera/protcastshared/ProtCast:$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

echo "============================================"
echo "Dual-encoder pilot: ${VARIANT} order, level ${LEVEL}, seed ${SEED}, pool=${POOL:-mean}"
echo "Features: ${FEATURE_ALGORITHMS}"
echo "  embeddings: $DATADIR/$EMBEDDIR-${LEVEL}${POOL_SUFFIX}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/compare_knn_vs_multilabel.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/$EMBEDDIR-${LEVEL}${POOL_SUFFIX} \
-o $OUTDIR \
--seed $SEED \
--order \
--order-variant $VARIANT \
--dual-encoder \
--feature_algorithms ${FEATURE_ALGORITHMS} \
${SCALE_ARG} \
--use_mlflow \
2>&1 | tee compare_dual_encoder_${VARIANT}_${POOL:-mean}_level_${LEVEL}_seed_${SEED}.log
