#!/bin/bash
#SBATCH --job-name run_knn_esm_vs_combined_ml
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_knn_esm_vs_combined_ml.out
#SBATCH -e run_knn_esm_vs_combined_ml.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 04:00:00

# Multi-level variant of run_compare_knn_esm_vs_knn_combined.sh.
# One SLURM job iterates GO depths in series for a single seed. Compared to
# submitting one job per (level, seed) this keeps total SLURM-job count low
# (good for staying under per-user queue caps) at the cost of serialising
# levels for that seed. Re-running is safe: each level's results.json is
# resumed independently by the inner Python script.
#
# Override LEVELS / SEED / FEATURE_ALGORITHMS / TAG / OUTROOT via --export
# from a launcher (see launch_psekraac_sweep_all_levels_per_seed.sh).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVELS=${LEVELS:-"5 6 7 8"}
SEED=${SEED:-42}
FEATURE_ALGORITHMS=${FEATURE_ALGORITHMS:-"CTriad Moran CTDD"}
TAG=${TAG:-default}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

for LEVEL in ${LEVELS}; do
    OUTDIR=${OUTROOT}/knn_esm_vs_combined-${TAG}-level-${LEVEL}-seed-${SEED}
    echo "============================================"
    echo "Running KNN(ESM-C) vs KNN(ESM-C + classical) for GO level ${LEVEL} (seed ${SEED})"
    echo "============================================"
    singularity exec --nv $CONTAINER \
    python3 scripts/compare_knn_esm_vs_knn_combined.py \
    -v \
    -p $DATADIR/ProtCastDataset.bin \
    -d $DATADIR/$EMBEDDIR-${LEVEL} \
    -o $OUTDIR \
    --feature_algorithms ${FEATURE_ALGORITHMS} \
    --seed $SEED \
    --use_mlflow \
    2>&1 | tee knn_esm_vs_combined_${TAG}_level_${LEVEL}_seed_${SEED}.log
done
