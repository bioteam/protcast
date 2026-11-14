#!/bin/bash
# Run sLAM
#SBATCH --job-name run-slam
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_slam-gpu.out
#SBATCH -e run_slam-gpu.err
#SBATCH -n 32
#SBATCH -N 2

CONTAINER=$WORK/tensorflow_2.17.0-gpu.sif

module load tacc-apptainer

export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages

export TF_FORCE_GPU_ALLOW_GROWTH=true

export GIT_PYTHON_REFRESH=silent

# Directory for Hugging Face datesets module
mkdir -p ${WORK}/HF
export HF_HOME=${WORK}/HF

cd ${HOME}/git/sLAM

singularity exec --nv $CONTAINER \
python3 sLAM/make-slam.py \
--epochs 30 \
-p "This is a test of" \
--num_datasets 10000 \
-v

# 10 epochs: 21:11:25
# 30 epochs: 
