#!/bin/bash
#SBATCH --job-name run_make_multi_class_model_embeds
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_make_multi_class_model_embds.out
#SBATCH -e run_make_multi_class_model_embeds.err
#SBATCH -N 2
#SBATCH -n 32

CONTAINER=${HOME}/tensorflow_2.17.0-gpu.sif
DATADIR=${WORK}/ProtCast/ProtCastDataset/11-03-2025

source ${HOME}/.bash_profile
# Only use modules from the container
unset PYTHONPATH
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

singularity exec --nv $CONTAINER \
python3 scripts/make_multi_class_model.py \
-v \
-p $DATADIR/ProtCastDataset.bin \
-d $DATADIR/mf_go_terms-level-4
