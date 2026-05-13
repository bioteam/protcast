#!/bin/bash
# Submit the KNN(ESM-C) vs KNN(ESM-C + PseKRAAC) comparison across seeds.
# Each seed becomes its own SLURM job, runs independently on rtx-dev, and
# writes to a per-seed output directory. Re-running is safe: each job
# resumes from any existing results JSON in its OUTDIR.
#
# Usage (from the ProtCast repo root on Frontera):
#     bash scripts/sh/launch_psekraac_sweep.sh
#
# Tweak LEVEL / SEEDS / FEATURES / TAG below as needed.

set -euo pipefail

LEVEL=4
SEEDS=(42 43 44)
FEATURES="PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"
TAG=psekraac

for SEED in "${SEEDS[@]}"; do
    sbatch \
        --export=ALL,SEED=${SEED},TAG=${TAG},FEATURE_ALGORITHMS="${FEATURES}",LEVEL=${LEVEL} \
        --job-name=knn_${TAG}_s${SEED} \
        -o run_knn_${TAG}_s${SEED}.out \
        -e run_knn_${TAG}_s${SEED}.err \
        protcastshared/ProtCast/scripts/sh/run_compare_knn_esm_vs_knn_combined.sh
done
