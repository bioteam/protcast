#!/bin/bash
#SBATCH --job-name run_feature_search
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_feature_search.out
#SBATCH -e run_feature_search.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 06:00:00

# Multi-level SLURM batch for the ORDER dual-encoder feature search.
# For one seed, iterates the given GO levels and, at each, trains the ESM-only
# order NN once then ranks candidate descriptor sets by the dual-encoder ΔFmax
# they produce. Tests whether per-level-optimal features let the order NN
# capture the all-level PseKRAAC signal KNN/flat already show.
#
# Env: LEVELS SEED VARIANT FEATURE_SETS OUTROOT
#
# No `set -u`: `module load` sources lmod/apptainer bash-completion that
# references unbound variables, which is fatal under nounset and would kill the
# job before python runs. The existing batch scripts omit `set` for this reason.

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVELS=${LEVELS:-"4 5 6 7 8"}
SEED=${SEED:-42}
VARIANT=${VARIANT:-soft}
# Candidate descriptors — the Doc-2 per-level winners plus the current fixed
# combo as a reference. Combine algorithms with '+'. Override via env.
FEATURE_SETS=${FEATURE_SETS:-"PseKRAAC_type_3B PseKRAAC_type_10 PseKRAAC_type_7 PseKRAAC_type_8 PseKRAAC_type_2 PseKRAAC_type_7+PseKRAAC_type_3B+PseKRAAC_type_8"}
OUTROOT=${OUTROOT:-${WORK}/ProtCast_results}

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

for LEVEL in ${LEVELS}; do
    OUTDIR=${OUTROOT}/dual_feature_search-level-${LEVEL}-seed-${SEED}-${VARIANT}
    echo "============================================"
    echo "Feature search: level ${LEVEL} seed ${SEED} (${VARIANT} order)"
    echo "============================================"
    singularity exec --nv $CONTAINER \
    python3 scripts/search_dual_encoder_features.py \
    -v \
    -p $DATADIR/ProtCastDataset.bin \
    -d $DATADIR/$EMBEDDIR-${LEVEL} \
    -o $OUTDIR \
    --seed $SEED \
    --order-variant $VARIANT \
    --feature-sets ${FEATURE_SETS} \
    2>&1 | tee feature_search_${VARIANT}_level_${LEVEL}_seed_${SEED}.log
done
