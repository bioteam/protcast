#!/bin/bash
#SBATCH --job-name run_dual_hparam
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_dual_hparam.out
#SBATCH -e run_dual_hparam.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

# Multi-level SLURM batch for the dual-encoder hyper-parameter sweep.
# Runs the 4-way comparison (KNN / flat / order / dual) plus the optional
# shuffled-FV capacity control, for one seed across the given levels, with
# whatever architecture/optimisation overrides are exported. Every override is
# opt-in: unset env vars => the flag is omitted => current behaviour reproduced.
#
# Env knobs (all optional except SEED via the launcher):
#   LEVELS FEATURE_ALGORITHMS VARIANT
#   FV_HIDDEN FV_DROPOUT GATED(0/1) ORDER_WEIGHT ORDER_DIM
#   LR LR_SCHEDULE PATIENCE MIN_DELTA SHUFFLE(0/1) OUTROOT
#
# Resume-safe: each level's results JSON skips already-completed arms, so the
# shuffled control can be added to an existing run without retraining the rest.
#
# No `set -u`: `module load` sources lmod/apptainer bash-completion that trips
# nounset and would kill the job before python runs. The ${VAR:-} guards below
# keep every optional knob safe without it.

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
# Pooling suffix on the embedding directory. Empty POOL -> legacy plain layout.
POOL=${POOL:-mean_max_std}
LEVELS=${LEVELS:-"6 7"}
SEED=${SEED:-42}
VARIANT=${VARIANT:-soft}
FEATURE_ALGORITHMS=${FEATURE_ALGORITHMS:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}
POOL_SUFFIX=${POOL:+-${POOL}}

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

# Build the opt-in flag list + an output-dir tag that encodes the swept knobs
# so different configurations don't overwrite each other's results.
EXTRA_ARGS=""
TAG="${VARIANT}order-dual"
[ -n "${FV_HIDDEN:-}" ]    && { EXTRA_ARGS="$EXTRA_ARGS --fv-hidden ${FV_HIDDEN}";       TAG="${TAG}-fvh${FV_HIDDEN}"; }
[ -n "${FV_DROPOUT:-}" ]   && { EXTRA_ARGS="$EXTRA_ARGS --fv-dropout ${FV_DROPOUT}";      TAG="${TAG}-fvd${FV_DROPOUT}"; }
[ -n "${ORDER_WEIGHT:-}" ] && { EXTRA_ARGS="$EXTRA_ARGS --order-weight ${ORDER_WEIGHT}";  TAG="${TAG}-ow${ORDER_WEIGHT}"; }
[ -n "${ORDER_DIM:-}" ]    && { EXTRA_ARGS="$EXTRA_ARGS --order-dim ${ORDER_DIM}";        TAG="${TAG}-od${ORDER_DIM}"; }
[ -n "${LR:-}" ]           && { EXTRA_ARGS="$EXTRA_ARGS --learning-rate ${LR}";           TAG="${TAG}-lr${LR}"; }
[ -n "${LR_SCHEDULE:-}" ]  && { EXTRA_ARGS="$EXTRA_ARGS --lr-schedule ${LR_SCHEDULE}";     TAG="${TAG}-${LR_SCHEDULE}"; }
[ -n "${PATIENCE:-}" ]     && { EXTRA_ARGS="$EXTRA_ARGS --patience ${PATIENCE}";           TAG="${TAG}-pat${PATIENCE}"; }
[ -n "${MIN_DELTA:-}" ]    && EXTRA_ARGS="$EXTRA_ARGS --min-delta ${MIN_DELTA}"
[ "${GATED:-0}" = "1" ]    && { EXTRA_ARGS="$EXTRA_ARGS --gated-fusion";                   TAG="${TAG}-gated"; }
[ "${SHUFFLE:-0}" = "1" ]  && EXTRA_ARGS="$EXTRA_ARGS --shuffle-fv-control"

echo "Extra args: ${EXTRA_ARGS:-<none>}"

for LEVEL in ${LEVELS}; do
    OUTDIR=${OUTROOT}/knn_vs_multilabel-${POOL:-mean}-level-${LEVEL}-seed-${SEED}-${TAG}
    echo "============================================"
    echo "Dual hparam: level ${LEVEL} seed ${SEED} pool=${POOL:-mean} tag ${TAG}"
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
    --use_mlflow \
    ${EXTRA_ARGS} \
    2>&1 | tee compare_dual_hparam_${TAG}_level_${LEVEL}_seed_${SEED}.log
done
