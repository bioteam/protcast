#!/bin/bash
# Submit the KNN(ESM-C) vs KNN(ESM-C + PseKRAAC) comparison across every
# combination of (level, seed). Each (level, seed) pair becomes its own
# SLURM job with its own output directory, so jobs are independent and
# re-running is safe — each job resumes from any results JSON already on
# disk in its OUTDIR.
#
# Routing: jobs go to `rtx` (production queue, 8-job per-user limit) with
# a 4-hour wall-clock. `rtx-dev` only allows 2 concurrent jobs and is
# unsuitable for a 15-job sweep.
#
# Usage (from the ProtCast repo root on Frontera):
#     bash scripts/sh/launch_psekraac_sweep_all_levels.sh
#
# Tweak LEVELS / SEEDS / FEATURES / TAG below to adjust the sweep.

set -euo pipefail

LEVELS=(5 6 7 8)
SEEDS=(42 43 44)
FEATURES="PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"
TAG=psekraac
PARTITION=rtx
WALLTIME=04:00:00

for LEVEL in "${LEVELS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        sbatch \
            -p ${PARTITION} \
            -t ${WALLTIME} \
            --export=ALL,SEED=${SEED},TAG=${TAG},FEATURE_ALGORITHMS="${FEATURES}",LEVEL=${LEVEL} \
            --job-name=knn_${TAG}_L${LEVEL}_s${SEED} \
            -o run_knn_${TAG}_L${LEVEL}_s${SEED}.out \
            -e run_knn_${TAG}_L${LEVEL}_s${SEED}.err \
            protcastshared/ProtCast/scripts/sh/run_compare_knn_esm_vs_knn_combined.sh
    done
done
