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

# Three-way comparison: KNN vs MultiLabel flat vs MultiLabel+box on ESM-C
# embeddings.  Adjust LEVEL to run different GO depths.  For levels deeper
# than 4, switch -p to rtx and increase -t (budget ~1hr per 50K proteins).

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
LEVEL=4
SEED=42
OUTDIR=${WORK}/ProtCast/knn_vs_multilabel-level-${LEVEL}-seed-${SEED}

# Only use local modules for Python 3.11 to match the Python version in the container
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

echo "============================================"
echo "Running KNN vs MultiLabel for GO level ${LEVEL}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/compare_knn_vs_multilabel.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/$EMBEDDIR-${LEVEL} \
-o $OUTDIR \
--seed $SEED \
--box \
--use_mlflow \
2>&1 | tee knn_vs_multilabel_level_${LEVEL}.log
