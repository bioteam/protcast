#!/bin/bash
#SBATCH --job-name run_knn_vs_multilabel
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_knn_vs_multilabel.out
#SBATCH -e run_knn_vs_multilabel.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

# Three-way comparison: KNN vs MultiLabel flat vs MultiLabel+order on ESM-C
# embeddings.  Adjust LEVEL to run different GO depths.  For levels deeper
# than 4, switch -p to rtx and increase -t (budget ~1hr per 50K proteins).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
# Pooling suffix on the embedding directory: mf_go_terms-level-<N>-<POOL>.
# Empty POOL falls back to the legacy plain mf_go_terms-level-<N> layout.
POOL=${POOL:-mean_max_std}
LEVEL=${LEVEL:-4}
SEED=${SEED:-42}
POOL_SUFFIX=${POOL:+-${POOL}}
# NN inputs are always standardized inside the driver (no flag needed).
OUTDIR=${OUTDIR:-${WORK}/ProtCast_results/knn_vs_multilabel-${POOL:-mean}-level-${LEVEL}-seed-${SEED}}

# Only use local modules for Python 3.11 to match the Python version in the container
# Prepend the repo root so `import protcast` resolves to THIS checkout, not a
# stale pip-installed copy in ~/.local (scripts are run by path, so the repo
# root is otherwise never on sys.path). Prevents new protcast API — e.g.
# MultiLabelClassifier(scale_features=...) — from silently hitting old code.
export PYTHONPATH=/work2/10504/wisdawg/frontera/protcastshared/ProtCast:$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

echo "============================================"
echo "Running KNN vs MultiLabel for GO level ${LEVEL} (pool=${POOL:-mean}, seed=${SEED})"
echo "  embeddings: $DATADIR/$EMBEDDIR-${LEVEL}${POOL_SUFFIX}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/compare_knn_vs_multilabel.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/$EMBEDDIR-${LEVEL}${POOL_SUFFIX} \
-o $OUTDIR \
--seed $SEED \
--order \
--use_mlflow \
2>&1 | tee knn_vs_multilabel_${POOL:-mean}_level_${LEVEL}_seed_${SEED}.log
