#!/bin/bash
#SBATCH --job-name run_make_multi_class_model_embeds
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_make_multi_class_model_embeds.out
#SBATCH -e run_make_multi_class_model_embeds.err
#SBATCH -N 2
#SBATCH -n 32

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/11-03-2025
EMBEDDIR=mf_go_terms-level-3

# Only use modules from the container with Python 3.11
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

singularity exec --nv $CONTAINER \
python3 scripts/make_multi_class_model_embeds.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
--input_source esm_embeddings \
-d $DATADIR/$EMBEDDIR
