#!/bin/bash
# Submit one SLURM job per seed for the ORDER dual-encoder feature search.
# Each job iterates all GO levels 4-8 in series, ranking candidate descriptors
# by the dual-encoder ΔFmax they yield over the ESM-only order baseline.
# Default: 3 jobs (seeds 42, 43, 44).
#
# This is the direct test of whether the order NN can capture PseKRAAC signal
# at every level (as KNN/flat do) once given the per-level-optimal descriptor.
#
# Usage (from the ProtCast repo root on Frontera):
#     bash scripts/sh/launch_feature_search.sh
#
# Override:
#     LEVELS="6 7" SEEDS_OVERRIDE="42" bash scripts/sh/launch_feature_search.sh

set -euo pipefail

LEVELS=${LEVELS:-"4 5 6 7 8"}
SEEDS=(${SEEDS_OVERRIDE:-42 43 44})
VARIANT=${VARIANT:-soft}

for SEED in "${SEEDS[@]}"; do
    sbatch \
        --export=ALL,SEED=${SEED},LEVELS="${LEVELS}",VARIANT=${VARIANT} \
        --job-name=featsearch_s${SEED} \
        -o run_featsearch_s${SEED}.out \
        -e run_featsearch_s${SEED}.err \
        protcastshared/ProtCast/scripts/sh/run_feature_search.sh
done
