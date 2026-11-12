#!/bin/bash
#SBATCH -J eval_perterm
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o eval_perterm_%x.out
#SBATCH -e eval_perterm_%x.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net

ALGO=${ALGO:-GAAC}
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026
WORKDIR=${WORK}/ProtCast
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast

echo "ALGO=$ALGO"
for POOLING in mean mean_max_std; do
  echo "######## POOLING=$POOLING ########"
  singularity exec $CONTAINER python3 $HOME/sh/eval_per_term.py \
    --pooling $POOLING --level 5 --algo $ALGO \
    --datadir $DATADIR --workdir $WORKDIR \
    --seeds 1 5 22 42 555 5555 \
    --out $WORKDIR/perterm_${ALGO}_L5_${POOLING}.csv
done
