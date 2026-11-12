#!/bin/bash
#SBATCH -J as_esm_fv
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 02:00:00
#SBATCH -o as_esm_fv.out
#SBATCH -e as_esm_fv.err
#SBATCH -A MCB22067
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
WORK_DIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026
DATASET=${WORK_DIR}/ProtCastDataset.bin
mkdir -p ${WORK}/HF
export HF_HOME=${WORK}/HF
source ${HOME}/.bash_profile
unset PYTHONPATH
module load tacc-apptainer
cd ${HOME}/git/ProtCast
singularity exec --nv $CONTAINER python3 $HOME/sh/make_active_site_esm_fv.py \
  -p $DATASET \
  --sites ${WORK}/ProtCast/uniprot_sites.json \
  --residues act_binding \
  -o ${WORK}/ProtCast/active_site_esm_fv.npz
