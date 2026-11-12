#!/bin/bash
#SBATCH -J ortho
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -o ortho_%x.out
#SBATCH -e ortho_%x.err

STAGE=${STAGE:?set STAGE=prep|analyze}
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
OUTDIR=${WORK}/ProtCast/structures
EMBDIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026/mf_go_terms-level-5-mean_max_std
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast

if [ "$STAGE" = "prep" ]; then
  singularity exec $CONTAINER python3 $HOME/sh/ortho_stage.py prep $OUTDIR $EMBDIR 30000
else
  singularity exec $CONTAINER python3 $HOME/sh/ortho_stage.py analyze $OUTDIR
fi
