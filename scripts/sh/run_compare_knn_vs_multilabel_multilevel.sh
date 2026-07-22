#!/bin/bash
#SBATCH --job-name run_knn_vs_multilabel_ml
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_knn_vs_multilabel_ml.out
#SBATCH -e run_knn_vs_multilabel_ml.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00

# Multi-level variant of run_compare_knn_vs_multilabel.sh. One SLURM job
# iterates GO depths in series for a single seed, running the 3-way
# comparison (KNN vs MultiLabel-flat vs MultiLabel+order) on ESM-C embeddings.
#
# NOTE: arm 3 is the order-embedding model (--order/--order-variant), which
# replaced the earlier box model after the order-embeddings merge. VARIANT is
# the order-violation loss variant: "soft" (default) or "hard".
#
# Unlike the KNN-only PseKRAAC sweep, this trains two neural models per level
# (flat + order, EPOCHS from config.json), so it goes to `rtx` with a 24h wall
# clock rather than rtx-dev/4h. Re-running is safe: each level's results.json
# is resumed independently, and completed arms are skipped.
#
# Override LEVELS / SEED / POOL / VARIANT / OUTROOT via --export from a
# launcher (see launch_multilabel_sweep_all_levels_per_seed.sh).
#
# ── Order-model tuning knobs (opt-in) ──────────────────────────────────────
# All OPT-IN. Leave a var empty (the default) and the corresponding CLI flag is
# not passed, so the driver falls back to config.json and reproduces prior runs
# bit-for-bit. Set any of them to tune the order model on the wider mean_max_std
# embeddings: ORDER_WEIGHT, ORDER_DIM, LEARNING_RATE, LR_SCHEDULE (only "cosine"),
# PATIENCE, MIN_DELTA.
#
# For the dual-encoder (ESM + PseKRAAC two-branch) variant use the dedicated
# toolchain instead: run_compare_dual_encoder{,_multilevel}.sh and
# run_dual_encoder_hparam.sh (all now POOL-aware).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
# Pooling suffix on the embedding directory. Empty POOL -> legacy plain layout.
POOL=${POOL:-mean_max_std}
LEVELS=${LEVELS:-"5 6 7 8"}
SEED=${SEED:-42}
VARIANT=${VARIANT:-soft}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}

# Tuning knobs (empty = use config.json default; flag omitted).
ORDER_WEIGHT=${ORDER_WEIGHT:-}
ORDER_DIM=${ORDER_DIM:-}
LEARNING_RATE=${LEARNING_RATE:-}
LR_SCHEDULE=${LR_SCHEDULE:-}
PATIENCE=${PATIENCE:-}
MIN_DELTA=${MIN_DELTA:-}

# Build the "-<POOL>" suffix only when POOL is non-empty.
POOL_SUFFIX=${POOL:+-${POOL}}

# Assemble the opt-in argument list. Only non-empty knobs contribute a flag, so
# a bare invocation is byte-identical to the previous bare "--order" runner.
EXTRA_ARGS=()
[ -n "$ORDER_WEIGHT" ]  && EXTRA_ARGS+=(--order-weight "$ORDER_WEIGHT")
[ -n "$ORDER_DIM" ]     && EXTRA_ARGS+=(--order-dim "$ORDER_DIM")
[ -n "$LEARNING_RATE" ] && EXTRA_ARGS+=(--learning-rate "$LEARNING_RATE")
[ -n "$LR_SCHEDULE" ]   && EXTRA_ARGS+=(--lr-schedule "$LR_SCHEDULE")
[ -n "$PATIENCE" ]      && EXTRA_ARGS+=(--patience "$PATIENCE")
[ -n "$MIN_DELTA" ]     && EXTRA_ARGS+=(--min-delta "$MIN_DELTA")

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

echo "Extra args: ${EXTRA_ARGS[*]:-<none, config.json defaults>}"

for LEVEL in ${LEVELS}; do
    EMBED_PATH=$DATADIR/$EMBEDDIR-${LEVEL}${POOL_SUFFIX}
    # Naming mirrors the existing "-softorder" result convention, with the pool
    # tag inserted so mean_max_std runs never clobber the old mean-pooled ones.
    OUTDIR=${OUTROOT}/knn_vs_multilabel-${POOL:-mean}-level-${LEVEL}-seed-${SEED}-${VARIANT}order
    echo "============================================"
    echo "KNN vs MultiLabel(flat + ${VARIANT} order) — GO level ${LEVEL} (pool=${POOL:-mean}, seed ${SEED})"
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
    "${EXTRA_ARGS[@]}" \
    --use_mlflow \
    2>&1 | tee knn_vs_multilabel_${VARIANT}order_${POOL:-mean}_level_${LEVEL}_seed_${SEED}.log
done
