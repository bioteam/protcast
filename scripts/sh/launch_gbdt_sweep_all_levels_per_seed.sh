#!/bin/bash
# Submit one SLURM job per seed for the GBDT (LightGBM) third-leg sweep:
# gbdt (raw ESM-C) and gbdt_combined (raw ESM-C + PseKRAAC) appended to the
# existing KNN / flat NN / order NN result directories.
# Each job iterates levels 5-8 in series inside a single allocation.
# Default: 3 SLURM jobs covering 12 (level, seed) pairs.
#
# Run the level-6 / seed-42 pilot first:
#     LEVELS="6" SEED=42 sbatch protcastshared/ProtCast/scripts/sh/run_compare_gbdt_multilevel.sh
# and check the log for the "GBDT" banners, then the JSON for gbdt /
# gbdt_combined keys and the FV gain-share line.
#
# Usage (from /work2/10504/wisdawg/frontera on Frontera):
#     bash protcastshared/ProtCast/scripts/sh/launch_gbdt_sweep_all_levels_per_seed.sh
#
# Overrides:
#     SHUFFLE_FV=0 bash ...            # drop the shuffled-PseKRAAC capacity control
#     FEATURES="PseKRAAC_type_2" bash ...
#     GBDT_N_ESTIMATORS=500 GBDT_COLSAMPLE_BYTREE=0.2 bash ...

set -euo pipefail

LEVELS="5 6 7 8"
SEEDS=(42 43 44)
POOL=${POOL:-mean_max_std}
VARIANT=${VARIANT:-soft}
# Include the shuffled-FV capacity control by default — it separates "real
# PseKRAAC signal" from "extra input dimensions" in the combined − gbdt delta.
SHUFFLE_FV=${SHUFFLE_FV:-1}
FEATURES=${FEATURES:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}

for SEED in "${SEEDS[@]}"; do
    sbatch \
        --export=ALL,SEED=${SEED},LEVELS="${LEVELS}",POOL=${POOL},VARIANT=${VARIANT},SHUFFLE_FV=${SHUFFLE_FV},FEATURE_ALGORITHMS="${FEATURES}" \
        --job-name=gbdt_${POOL}_s${SEED} \
        -o run_gbdt_${POOL}_s${SEED}.out \
        -e run_gbdt_${POOL}_s${SEED}.err \
        protcastshared/ProtCast/scripts/sh/run_compare_gbdt_multilevel.sh
done
