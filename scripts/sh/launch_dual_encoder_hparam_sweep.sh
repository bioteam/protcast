#!/bin/bash
# Submit the dual-encoder hyper-parameter sweep: fv_hidden ∈ {8,16,32,64,128}
# across seeds {42,43,44} at the extension-relevant levels L6 (the known win)
# and L7 (the underpowered-but-positive candidate). The shuffled-FV capacity
# control is folded into the reference-width (fv_hidden=32) job only, so it runs
# once per (level, seed) without wasting compute or racing other jobs.
#
# Total: 3 seeds × 5 fv_hidden = 15 SLURM jobs, each iterating L6 & L7.
#
# Usage (from the ProtCast repo root on Frontera):
#     bash scripts/sh/launch_dual_encoder_hparam_sweep.sh
#
# Override the grid:
#     LEVELS="6" FV_HIDDENS="16 32 64" bash scripts/sh/launch_dual_encoder_hparam_sweep.sh

set -euo pipefail

LEVELS=${LEVELS:-"6 7"}
SEEDS=(42 43 44)
FV_HIDDENS=(${FV_HIDDENS:-8 16 32 64 128})
VARIANT=${VARIANT:-soft}
FEATURES=${FEATURES:-"PseKRAAC_type_7 PseKRAAC_type_3B PseKRAAC_type_8"}

for SEED in "${SEEDS[@]}"; do
    for FVH in "${FV_HIDDENS[@]}"; do
        # Run the capacity control once, at the reference width (fv_hidden=32).
        SHUF=0
        [ "$FVH" = "32" ] && SHUF=1
        sbatch \
            --export=ALL,SEED=${SEED},FV_HIDDEN=${FVH},SHUFFLE=${SHUF},LEVELS="${LEVELS}",VARIANT=${VARIANT},FEATURE_ALGORITHMS="${FEATURES}" \
            --job-name=dualhp_fvh${FVH}_s${SEED} \
            -o run_dualhp_fvh${FVH}_s${SEED}.out \
            -e run_dualhp_fvh${FVH}_s${SEED}.err \
            protcastshared/ProtCast/scripts/sh/run_dual_encoder_hparam.sh
    done
done
