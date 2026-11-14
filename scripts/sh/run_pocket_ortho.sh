#!/bin/bash
#SBATCH -J pocket_ortho
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -o pocket_ortho.out
#SBATCH -e pocket_ortho.err
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
STRUCTDIR=${WORK}/ProtCast/structures
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast
singularity exec $CONTAINER python3 $HOME/sh/ortho_npz.py $STRUCTDIR/sample_emb.npz $STRUCTDIR/pocket_fv.npz $STRUCTDIR/pocket_orthogonality_R2.csv
