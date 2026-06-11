#!/bin/bash
#SBATCH --job-name run_compare_order_combined_ml
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_compare_order_combined_ml.out
#SBATCH -e run_compare_order_combined_ml.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

# Multi-level variant of run_compare_order_combined.sh.
# One SLURM job iterates GO depths in series for a single seed, running the
# two-way feature comparison (Order NN on ESM vs Order NN on ESM+PseKRAAC) at
# each level. Compared to one job per (level, seed) this keeps total SLURM-job
# count low (good for staying under per-user queue caps) at the cost of
# serialising levels for that seed. Re-running is safe: each level's
# results.json is resumed independently by the inner Python script.
#
# Override LEVELS / SEED / VARIANT / FEATURE_ALGORITHMS / OUTROOT via --export
# from a launcher (see launch_order_combined_sweep_all_levels_per_seed.sh).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVELS=${LEVELS:-"5 6 7 8"}
SEED=${SEED:-42}
# Order-violation loss variant: "soft" (default, softplus) or "hard" (ReLU).
VARIANT=${VARIANT:-soft}
# Classical feature-vector algorithms concatenated with ESM for the combined arm.
FEATURE_ALGORITHMS=${FEATURE_ALGORITHMS:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

for LEVEL in ${LEVELS}; do
    OUTDIR=${OUTROOT}/order-esm-vs-combined-level-${LEVEL}-seed-${SEED}-${VARIANT}order
    echo "============================================"
    echo "Order ESM vs ESM+PseKRAAC: ${VARIANT} order, GO level ${LEVEL} (seed ${SEED})"
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
done
