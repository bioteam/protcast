#!/bin/bash
#SBATCH -J shape_train
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 02:00:00
#SBATCH -o shape_train.out
#SBATCH -e shape_train.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
EMBDIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026/mf_go_terms-level-5-mean_max_std
STRUCTDIR=${WORK}/ProtCast/structures
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast
singularity exec --nv $CONTAINER python3 $HOME/sh/shape_go_test.py $EMBDIR $STRUCTDIR 1,42,555,5555 $STRUCTDIR/shape_go_result.json
