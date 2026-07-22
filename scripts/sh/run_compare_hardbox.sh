#!/bin/bash
#SBATCH --job-name compare_hardbox
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o compare_hardbox.out
#SBATCH -e compare_hardbox.err
#SBATCH -p rtx-dev
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=/work2/04769/bosborne/frontera/ProtCast/ProtCastDataset/01-23-2026
EMBEDDIR=mf_go_terms-level
POOL=${POOL:-mean_max_std}
LEVEL=${LEVEL:-4}
SEED=${SEED:-42}
VARIANT=hard
POOL_SUFFIX=${POOL:+-${POOL}}
OUTDIR=${OUTDIR:-${WORK}/ProtCast_results/knn_vs_multilabel-${POOL:-mean}-level-${LEVEL}-seed-${SEED}-${VARIANT}box}

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd /work2/10504/wisdawg/frontera/protcastshared/ProtCast/

echo "============================================"
echo "Geometry comparison: ${VARIANT} box, level ${LEVEL}, seed ${SEED}, pool=${POOL:-mean}"
echo "============================================"
singularity exec --nv $CONTAINER \
python3 scripts/compare_knn_vs_multilabel.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/$EMBEDDIR-${LEVEL}${POOL_SUFFIX} \
-o $OUTDIR \
--seed $SEED \
--box \
--box-variant $VARIANT \
--use_mlflow \
2>&1 | tee compare_${VARIANT}box_${POOL:-mean}_level_${LEVEL}_seed_${SEED}.log
