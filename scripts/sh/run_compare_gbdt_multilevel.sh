#!/bin/bash
#SBATCH --job-name run_gbdt_ml
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_gbdt_ml.out
#SBATCH -e run_gbdt_ml.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00

# GBDT "third leg" of the KNN vs MultiLabel comparison. One SLURM job iterates
# GO depths in series for a single seed and adds the LightGBM arms to
# compare_knn_vs_multilabel.py:
#
#   gbdt                    LightGBM one-vs-rest on RAW ESM-C
#   gbdt_combined           same, on RAW ESM-C + PseKRAAC (reports ESM-vs-FV gain share)
#   gbdt_combined_shuffled  capacity control on permuted PseKRAAC (SHUFFLE_FV=1)
#
# Trees are scale-invariant, so no StandardScaler is involved — this arm is
# free of the block-scaling confound the neural arms must manage.
#
# RESUME BEHAVIOUR (the default): OUTDIR is the SAME directory the existing
# multilevel sweep wrote (…-${VARIANT}order), so the driver finds knn /
# multilabel_flat / multilabel_order already "ok" in the results JSON and trains
# ONLY the GBDT arms, appending them to that JSON. Set OUTSUFFIX=-gbdt to write
# a fresh directory instead (knn + flat [+ order] then retrain too).
#
# ORDER=1 (default) passes --order so the resume set matches the sweep dirs.
# ORDER=0 omits it (fresh dirs then get knn + flat + gbdt arms only).
#
# Requires lightgbm inside the container's user site:
#     module load tacc-apptainer
#     singularity exec ${WORK}/tensorflow_2.17.0-gpu.sif pip3 install --user lightgbm
#
# ── GBDT tuning knobs (opt-in) ─────────────────────────────────────────────
# Leave a var empty (the default) and the flag is not passed, so the driver
# falls back to config.json / class defaults: GBDT_N_ESTIMATORS (300),
# GBDT_LEARNING_RATE (0.05), GBDT_NUM_LEAVES (31), GBDT_MIN_CHILD_SAMPLES (20),
# GBDT_COLSAMPLE_BYTREE (0.3), GBDT_SUBSAMPLE (0.8), GBDT_REG_LAMBDA (1.0),
# GBDT_N_JOBS (0 = all cores), GBDT_EARLY_STOPPING_ROUNDS (0 = off).
#
# Override LEVELS / SEED / POOL / VARIANT / OUTROOT / FEATURE_ALGORITHMS via
# --export from a launcher (see launch_gbdt_sweep_all_levels_per_seed.sh).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
# Pooling suffix on the embedding directory. Empty POOL -> legacy plain layout.
POOL=${POOL:-mean_max_std}
LEVELS=${LEVELS:-"5 6 7 8"}
SEED=${SEED:-42}
VARIANT=${VARIANT:-soft}
ORDER=${ORDER:-1}
SHUFFLE_FV=${SHUFFLE_FV:-0}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}
OUTSUFFIX=${OUTSUFFIX:-}
FEATURE_ALGORITHMS=${FEATURE_ALGORITHMS:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}

# Tuning knobs (empty = use config.json / class default; flag omitted).
GBDT_N_ESTIMATORS=${GBDT_N_ESTIMATORS:-}
GBDT_LEARNING_RATE=${GBDT_LEARNING_RATE:-}
GBDT_NUM_LEAVES=${GBDT_NUM_LEAVES:-}
GBDT_MIN_CHILD_SAMPLES=${GBDT_MIN_CHILD_SAMPLES:-}
GBDT_COLSAMPLE_BYTREE=${GBDT_COLSAMPLE_BYTREE:-}
GBDT_SUBSAMPLE=${GBDT_SUBSAMPLE:-}
GBDT_REG_LAMBDA=${GBDT_REG_LAMBDA:-}
GBDT_N_JOBS=${GBDT_N_JOBS:-}
GBDT_EARLY_STOPPING_ROUNDS=${GBDT_EARLY_STOPPING_ROUNDS:-}

# Build the "-<POOL>" suffix only when POOL is non-empty.
POOL_SUFFIX=${POOL:+-${POOL}}

# Assemble the argument list. --gbdt is always on; everything else is opt-in.
# shellcheck disable=SC2206
EXTRA_ARGS=(--gbdt --feature_algorithms ${FEATURE_ALGORITHMS})
[ "$ORDER" = "1" ]      && EXTRA_ARGS+=(--order --order-variant "$VARIANT")
[ "$SHUFFLE_FV" = "1" ] && EXTRA_ARGS+=(--shuffle-fv-control)
[ -n "$GBDT_N_ESTIMATORS" ]          && EXTRA_ARGS+=(--gbdt-n-estimators "$GBDT_N_ESTIMATORS")
[ -n "$GBDT_LEARNING_RATE" ]         && EXTRA_ARGS+=(--gbdt-learning-rate "$GBDT_LEARNING_RATE")
[ -n "$GBDT_NUM_LEAVES" ]            && EXTRA_ARGS+=(--gbdt-num-leaves "$GBDT_NUM_LEAVES")
[ -n "$GBDT_MIN_CHILD_SAMPLES" ]     && EXTRA_ARGS+=(--gbdt-min-child-samples "$GBDT_MIN_CHILD_SAMPLES")
[ -n "$GBDT_COLSAMPLE_BYTREE" ]      && EXTRA_ARGS+=(--gbdt-colsample-bytree "$GBDT_COLSAMPLE_BYTREE")
[ -n "$GBDT_SUBSAMPLE" ]             && EXTRA_ARGS+=(--gbdt-subsample "$GBDT_SUBSAMPLE")
[ -n "$GBDT_REG_LAMBDA" ]            && EXTRA_ARGS+=(--gbdt-reg-lambda "$GBDT_REG_LAMBDA")
[ -n "$GBDT_N_JOBS" ]                && EXTRA_ARGS+=(--gbdt-n-jobs "$GBDT_N_JOBS")
[ -n "$GBDT_EARLY_STOPPING_ROUNDS" ] && EXTRA_ARGS+=(--gbdt-early-stopping-rounds "$GBDT_EARLY_STOPPING_ROUNDS")

# Prepend the repo root so `import protcast` resolves to THIS checkout, not a
# stale pip-installed copy in ~/.local. ~/.local stays on the path because that
# is where `pip3 install --user lightgbm` lands.
export PYTHONPATH=/work2/10504/wisdawg/frontera/protcastshared/ProtCast:$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

echo "Extra args: ${EXTRA_ARGS[*]}"
singularity exec $CONTAINER python3 -c "import lightgbm; print('lightgbm', lightgbm.__version__)" \
    || { echo "ERROR: lightgbm not importable inside the container (see header for install)"; exit 1; }

for LEVEL in ${LEVELS}; do
    EMBED_PATH=$DATADIR/$EMBEDDIR-${LEVEL}${POOL_SUFFIX}
    # Same directory convention as run_compare_knn_vs_multilabel_multilevel.sh so
    # completed knn / flat / order arms are resumed rather than retrained.
    OUTDIR=${OUTROOT}/knn_vs_multilabel-${POOL:-mean}-level-${LEVEL}-seed-${SEED}-${VARIANT}order${OUTSUFFIX}
    echo "============================================"
    echo "GBDT arms (LightGBM) — GO level ${LEVEL} (pool=${POOL:-mean}, seed ${SEED})"
    echo "  embeddings: ${EMBED_PATH}"
    echo "  output:     ${OUTDIR}"
    echo "============================================"
    singularity exec --nv $CONTAINER \
    python3 scripts/compare_knn_vs_multilabel.py \
    -v \
    -p $DATADIR/ProtCastDataset.bin \
    -d $EMBED_PATH \
    -o $OUTDIR \
    --seed $SEED \
    "${EXTRA_ARGS[@]}" \
    --use_mlflow \
    2>&1 | tee knn_vs_multilabel_gbdt_${POOL:-mean}_level_${LEVEL}_seed_${SEED}.log
done
