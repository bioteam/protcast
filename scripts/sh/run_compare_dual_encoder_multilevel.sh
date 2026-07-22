#!/bin/bash
#SBATCH --job-name run_dual_encoder_ml
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_dual_encoder_ml.out
#SBATCH -e run_dual_encoder_ml.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

# Multi-level SLURM batch script for the dual-encoder order NN sweep.
# One job iterates GO depths in series for a single seed. Compared to
# submitting one job per (level, seed) this stays under the per-user queue
# cap. Re-running is safe: each level's results JSON is resumed independently.
#
# Override LEVELS / SEED / POOL / VARIANT / FEATURE_ALGORITHMS / OUTROOT via
# --export from the launcher (see launch_dual_encoder_sweep_all_levels_per_seed.sh).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
# Pooling suffix on the embedding directory: mf_go_terms-level-<N>-<POOL>.
# Empty POOL falls back to the legacy plain mf_go_terms-level-<N> layout.
POOL=${POOL:-mean_max_std}
LEVELS=${LEVELS:-"5 6 7 8"}
SEED=${SEED:-42}
VARIANT=${VARIANT:-soft}
FEATURE_ALGORITHMS=${FEATURE_ALGORITHMS:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}
# Scale the single-encoder baseline arms (flat/order); no-op for the dual arm,
# which already scales its blocks. MANDATORY for mean_max_std.
SCALE_FEATURES=${SCALE_FEATURES:-1}
SCALE_ARG=""; [ "$SCALE_FEATURES" = "1" ] && SCALE_ARG="--scale-features"
# Capacity control: also train a dual encoder on PseKRAAC vectors shuffled
# across proteins. If dual gains over the ESM-only order baseline but the
# shuffled arm gains just as much, the "gain" is added capacity, not FV signal.
# Adds one extra model per level; results land in the same results.json.
SHUFFLE_FV=${SHUFFLE_FV:-0}
SHUFFLE_ARG=""; [ "$SHUFFLE_FV" = "1" ] && SHUFFLE_ARG="--shuffle-fv-control"

POOL_SUFFIX=${POOL:+-${POOL}}

# Prepend the repo root so `import protcast` resolves to THIS checkout, not a
# stale pip-installed copy in ~/.local (scripts are run by path, so the repo
# root is otherwise never on sys.path). Prevents new protcast API — e.g.
# MultiLabelClassifier(scale_features=...) — from silently hitting old code.
export PYTHONPATH=/work2/10504/wisdawg/frontera/protcastshared/ProtCast:$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

for LEVEL in ${LEVELS}; do
    EMBED_PATH=$DATADIR/$EMBEDDIR-${LEVEL}${POOL_SUFFIX}
    OUTDIR=${OUTROOT}/knn_vs_multilabel-${POOL:-mean}-level-${LEVEL}-seed-${SEED}-${VARIANT}order-dual
    echo "============================================"
    echo "Dual-encoder: ${VARIANT} order, GO level ${LEVEL} (pool=${POOL:-mean}, seed ${SEED})"
    echo "  embeddings: ${EMBED_PATH}"
    echo "============================================"
    singularity exec --nv $CONTAINER \
    python3 scripts/compare_knn_vs_multilabel.py \
    -v \
    -p $DATADIR/ProtCastDataset.bin \
    -d $EMBED_PATH \
    -o $OUTDIR \
    --seed $SEED \
    --order \
    --order-variant $VARIANT \
    --dual-encoder \
    --feature_algorithms ${FEATURE_ALGORITHMS} \
    ${SCALE_ARG} \
    ${SHUFFLE_ARG} \
    --use_mlflow \
    2>&1 | tee compare_dual_encoder_${VARIANT}_${POOL:-mean}_level_${LEVEL}_seed_${SEED}.log
done
