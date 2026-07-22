#!/bin/bash
# Submit the KNN vs MultiLabel(flat + box) 3-way comparison on mean_max_std
# embeddings. One SLURM job per seed; each job iterates levels 5-8 in series.
# Total: 3 jobs, under the rtx 8-per-user submit cap, covering 12 (level, seed)
# pairs.
#
# This is the neural counterpart to launch_psekraac_sweep_all_levels_per_seed.sh.
# Because it trains flat + box NNs per level it uses `rtx` with a 24h wall clock
# (set inside run_compare_knn_vs_multilabel_multilevel.sh), not rtx-dev.
#
# Re-running is safe: per-level results.json files are resumed by the Python
# driver, so an interrupted or timed-out job picks up where it left off.
#
# Usage (from /work2/10504/wisdawg/frontera on Frontera — the sbatch target
# below is a path relative to that directory, matching the PseKRAAC launcher):
#     bash protcastshared/ProtCast/scripts/sh/launch_multilabel_sweep_all_levels_per_seed.sh
#
# SMOKE TEST FIRST. Neural wall-clock on 3456-dim mean_max_std embeddings is
# unmeasured. Before committing the full grid, time one (level, seed):
#     POOL=mean_max_std LEVEL=5 SEED=42 \
#       sbatch -p rtx-dev -t 02:00:00 \
#       protcastshared/ProtCast/scripts/sh/run_compare_knn_vs_multilabel.sh
# Then scale the walltime in the multilevel runner from what you observe.

set -euo pipefail

LEVELS="5 6 7 8"
SEEDS=(42 43 44)
POOL=${POOL:-mean_max_std}
VARIANT=${VARIANT:-hard}
TAG=mlbox

for SEED in "${SEEDS[@]}"; do
    sbatch \
        --export=ALL,SEED=${SEED},LEVELS="${LEVELS}",POOL=${POOL},VARIANT=${VARIANT},TAG=${TAG} \
        --job-name=mlbox_${POOL}_ml_s${SEED} \
        -o run_mlbox_${POOL}_ml_s${SEED}.out \
        -e run_mlbox_${POOL}_ml_s${SEED}.err \
        protcastshared/ProtCast/scripts/sh/run_compare_knn_vs_multilabel_multilevel.sh
done
