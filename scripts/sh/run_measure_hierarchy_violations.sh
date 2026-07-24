#!/bin/bash
#SBATCH --job-name run_measure_hierarchy_violations
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_measure_hierarchy_violations.out
#SBATCH -e run_measure_hierarchy_violations.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00

# Measure GO-DAG true-path-rule violations for the trained models, to quantify
# how much "hierarchy signal is on the table" (see scripts/measure_hierarchy_violations.py).
# Inference only — loads saved models, predicts on the reconstructed validation
# split, counts violations, and reports the true-path-rule delta Fmax per arm.
#
# Two passes in one job:
#   1. Depth scan across LEVELS using the Tier-1 sweep dirs. KNN is scale-
#      independent so it self-checks (and reports) cleanly everywhere; the flat/
#      order arms in those PRE-scale-fix dirs will fail their self-check and
#      auto-skip (expected — their .keras were trained unscaled).
#   2. Full three-arm (KNN / flat / order) on the fixed-order smoke dir, where
#      flat and order were trained with the standardization + order fixes.
#
# Override LEVELS / SEED / POOL / OUTROOT / SMOKE_OUTDIR / SMOKE_LEVEL via env.

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
POOL=${POOL:-mean_max_std}
LEVELS=${LEVELS:-"5 6 7 8"}
SEED=${SEED:-42}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}
# Fixed-order three-arm dir (set SMOKE_OUTDIR="" to skip pass 2).
SMOKE_OUTDIR=${SMOKE_OUTDIR:-${OUTROOT}/orderfix-smoke-L6-s42}
SMOKE_LEVEL=${SMOKE_LEVEL:-6}

POOL_SUFFIX=${POOL:+-${POOL}}

# The Python script self-inserts the repo root, so it always loads the fixed
# protcast (incl. the corrected OrderEmbeddingLayer) even with this PYTHONPATH.
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

run_one () {   # $1 = level, $2 = results dir
    local level="$1" outdir="$2"
    if [ ! -d "$outdir" ]; then
        echo "SKIP level ${level}: results dir not found: ${outdir}"
        return
    fi
    echo "============================================"
    echo "Hierarchy violations — level ${level} (pool=${POOL})"
    echo "  results: ${outdir}"
    echo "============================================"
    singularity exec --nv $CONTAINER \
        python3 scripts/measure_hierarchy_violations.py \
        -d $DATADIR/$EMBEDDIR-${level}${POOL_SUFFIX} \
        -p $DATADIR/ProtCastDataset.bin \
        -o "$outdir" \
        --seed $SEED \
        -v
}

# ── Pass 1: depth scan over the Tier-1 sweep dirs (KNN across L5-8) ─────────
for LEVEL in ${LEVELS}; do
    run_one "$LEVEL" \
        "${OUTROOT}/knn_vs_multilabel-${POOL}-level-${LEVEL}-seed-${SEED}-softorder"
done

# ── Pass 2: full three-arm on the fixed-order smoke dir ────────────────────
if [ -n "$SMOKE_OUTDIR" ]; then
    run_one "$SMOKE_LEVEL" "$SMOKE_OUTDIR"
fi

echo "Done."
