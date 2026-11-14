#!/bin/bash
# Run sLAM
#SBATCH --job-name run-slam
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
#SBATCH -o run-slam-cpu.out
#SBATCH -e run-slam-cpu.err
#SBATCH -n 56
#SBATCH -N 1

source ${HOME}/.bash_profile
source ${HOME}/tf/bin/activate

cd ${HOME}/git/sLAM

#time python3 sLAM/get-readmes.py

time python3 sLAM/make-slam.py -d --epochs 1 -p "This is a test" -v -t 30
