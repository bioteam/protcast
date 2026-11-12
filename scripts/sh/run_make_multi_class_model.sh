#!/bin/bash
#SBATCH --job-name run_make_multi_class_model-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_make_multi_class_model-gpu.out
#SBATCH -e run_make_multi_class_model-gpu.err
#SBATCH -N 2
#SBATCH -n 32

CONTAINER=${HOME}/tensorflow_2.17.0-gpu.sif
ALGORITHM=K2TPC
DATADIR=${WORK}/ProtCast/ProtCastDataset/11-03-2025
GOFILE=mf_go_terms-level-4.tsv

source ${HOME}/.bash_profile
# Only use modules from the container
unset PYTHONPATH
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

singularity exec --nv $CONTAINER \
python3 scripts/make_multi_class_model.py \
-v \
--minimum_seqs 200 \
--maximum_seqs 1000 \
-p $DATADIR/ProtCastDataset.bin \
-a $ALGORITHM \
-g $DATADIR/$GOFILE
