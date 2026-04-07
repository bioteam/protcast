#!/bin/bash
#SBATCH --job-name run_make_esm_embeddings
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_make_esm_embeddings.out
#SBATCH -e run_make_esm_embeddings.err
#SBATCH -N 2
#SBATCH -n 32

MODEL_TYPE=esmc_600m
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
WORK_DIR=${WORK}/ProtCast/ProtCastDataset/11-03-2025/
DATASET=${WORK_DIR}/ProtCastDataset.bin
GO_FILE=${WORK_DIR}/mf_go_terms-level-4.tsv

source ${HOME}/.bash_profile
# Only use modules from the container
unset PYTHONPATH
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

singularity exec --nv $CONTAINER \
python3 scripts/make_esm_embeddings.py \
-v \
-p $DATASET \
-g $GO_FILE \
-o $WORK \
--model_type $MODEL_TYPE
