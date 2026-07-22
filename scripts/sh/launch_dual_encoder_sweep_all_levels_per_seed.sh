#!/bin/bash
# Submit one SLURM job per seed for the dual-encoder order NN sweep
# (KNN vs flat NN vs order NN vs order NN + dual-encoder PseKRAAC).
# Each job iterates levels 5-8 in series inside a single allocation.
# Default: 3 SLURM jobs covering 12 (level, seed) pairs.
#
# Only run this after the pilot (run_compare_dual_encoder.sh at level 6,
# seed 42) shows a meaningful Fmax improvement over the ESM-only order NN.
# If the pilot shows <0.010 Fmax gain the full sweep is not worth the
# compute — the single-encoder order NN is self-sufficient on ESM-C.
#
# Usage (from the ProtCast repo root on Frontera):
#     bash scripts/sh/launch_dual_encoder_sweep_all_levels_per_seed.sh
#
# Override variant or features:
#     VARIANT=hard bash scripts/sh/launch_dual_encoder_sweep_all_levels_per_seed.sh
#     FEATURES="PseKRAAC_type_2 PseKRAAC_type_5" bash ...

set -euo pipefail

LEVELS="5 6 7 8"
SEEDS=(42 43 44)
POOL=${POOL:-mean_max_std}
VARIANT=${VARIANT:-soft}
FEATURES=${FEATURES:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}

for SEED in "${SEEDS[@]}"; do
    sbatch \
        --export=ALL,SEED=${SEED},LEVELS="${LEVELS}",POOL=${POOL},VARIANT=${VARIANT},FEATURE_ALGORITHMS="${FEATURES}" \
        --job-name=dual_enc_${VARIANT}_${POOL}_s${SEED} \
        -o run_dual_enc_${VARIANT}_${POOL}_s${SEED}.out \
        -e run_dual_enc_${VARIANT}_${POOL}_s${SEED}.err \
        protcastshared/ProtCast/scripts/sh/run_compare_dual_encoder_multilevel.sh
done
