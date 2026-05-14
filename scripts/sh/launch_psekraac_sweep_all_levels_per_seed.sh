#!/bin/bash
# Submit one SLURM job per seed. Each job iterates levels 5-8 in series
# inside a single allocation. Total: 3 SLURM jobs, well under the rtx
# 8-per-user submit cap, covering 12 (level, seed) pairs.
#
# Why not one job per (level, seed): that's 12 jobs and hits the cap.
# Why not one job for everything: serialising all 12 in one allocation
# wastes parallelism — three parallel jobs each doing 4 levels finishes
# in roughly 1/3 the wall-clock of a single 12-step job.
#
# Re-running is safe: per-level results.json files are resumed by the
# Python script, so an interrupted job picks up where it left off.
#
# Usage (from the ProtCast repo root on Frontera):
#     bash scripts/sh/launch_psekraac_sweep_all_levels_per_seed.sh

set -euo pipefail

LEVELS="5 6 7 8"
SEEDS=(42 43 44)
FEATURES="PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"
TAG=psekraac

for SEED in "${SEEDS[@]}"; do
    sbatch \
        --export=ALL,SEED=${SEED},LEVELS="${LEVELS}",TAG=${TAG},FEATURE_ALGORITHMS="${FEATURES}" \
        --job-name=knn_${TAG}_ml_s${SEED} \
        -o run_knn_${TAG}_ml_s${SEED}.out \
        -e run_knn_${TAG}_ml_s${SEED}.err \
        protcastshared/ProtCast/scripts/sh/run_compare_knn_esm_vs_knn_combined_multilevel.sh
done
