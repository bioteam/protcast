#!/bin/bash
#SBATCH --job-name run_compare_order_ml
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_compare_order_ml.out
#SBATCH -e run_compare_order_ml.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

# Multi-level variant of run_compare_order.sh.
# One SLURM job iterates GO depths in series for a single seed, running the
# three-way comparison (KNN vs flat NN vs order-embedding NN) at each level.
# Compared to submitting one job per (level, seed) this keeps total SLURM-job
# count low (good for staying under per-user queue caps) at the cost of
# serialising levels for that seed. Re-running is safe: each level's
# results.json is resumed independently by the inner Python script.
#
# Override LEVELS / SEED / VARIANT / OUTROOT via --export from a launcher
# (see launch_order_sweep_all_levels_per_seed.sh).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVELS=${LEVELS:-"5 6 7 8"}
SEED=${SEED:-42}
# Order-violation loss variant: "soft" (default, softplus) or "hard" (ReLU).
VARIANT=${VARIANT:-soft}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

for LEVEL in ${LEVELS}; do
    OUTDIR=${OUTROOT}/knn_vs_multilabel-level-${LEVEL}-seed-${SEED}-${VARIANT}order
    echo "============================================"
    echo "Order comparison: ${VARIANT} order, GO level ${LEVEL} (seed ${SEED})"
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
done
