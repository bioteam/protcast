#!/bin/bash
#SBATCH -J shape_fv
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -o shape_fv.out
#SBATCH -e shape_fv.err
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast
singularity exec $CONTAINER python3 $HOME/sh/compute_shape_fv.py ${WORK}/ProtCast/structures
