#!/bin/bash
#SBATCH -J rand_esm_train
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 03:00:00
#SBATCH -o rand_esm_train.out
#SBATCH -e rand_esm_train.err
#SBATCH -A MCB22067
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
EMBDIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026/mf_go_terms-level-5-mean_max_std
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast
FV=${WORK}/ProtCast/rand_esm_fv.npz
singularity exec --nv $CONTAINER python3 $HOME/sh/fv_go_test.py $EMBDIR $FV 1,42,555,5555,5,22,7,13,99,2024,314,271 ${WORK}/ProtCast/rand_esm_go_result.json rand_esm
