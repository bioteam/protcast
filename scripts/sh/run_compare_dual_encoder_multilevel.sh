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
# Override LEVELS / SEED / VARIANT / FEATURE_ALGORITHMS / OUTROOT via
# --export from the launcher (see launch_dual_encoder_sweep_all_levels_per_seed.sh).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVELS=${LEVELS:-"5 6 7 8"}
SEED=${SEED:-42}
VARIANT=${VARIANT:-soft}
FEATURE_ALGORITHMS=${FEATURE_ALGORITHMS:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

for LEVEL in ${LEVELS}; do
    OUTDIR=${OUTROOT}/knn_vs_multilabel-level-${LEVEL}-seed-${SEED}-${VARIANT}order-dual
    echo "============================================"
    echo "Dual-encoder: ${VARIANT} order, GO level ${LEVEL} (seed ${SEED})"
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
    --dual-encoder \
    --feature_algorithms ${FEATURE_ALGORITHMS} \
    --use_mlflow \
    2>&1 | tee compare_dual_encoder_${VARIANT}_level_${LEVEL}_seed_${SEED}.log
done
