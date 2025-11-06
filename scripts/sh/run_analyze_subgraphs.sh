#!/bin/bash
#SBATCH --job-name run_analyze_subgraphs-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run_analyze_subgraphs-gpu.out
#SBATCH -e run_analyze_subgraphs-gpu.err
#SBATCH -N 2
#SBATCH -n 32

#DIR=multi-classification
#NAME=gpu
CONTAINER=~/tensorflow_2.17.0-gpu.sif
ALGORITHM=CTriad
DATASET=/work2/04769/bosborne/frontera/ProtCast/

source ${HOME}/.bash_profile
# Only use modules from the container
unset PYTHONPATH
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

singularity exec --nv $CONTAINER python3 scripts/analyze_subgraphs.py -v -p $DATASET -a $ALGORITHM