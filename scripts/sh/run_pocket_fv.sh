#!/bin/bash
#SBATCH -J pocket_fv
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -o pocket_fv.out
#SBATCH -e pocket_fv.err
module load python3 2>/dev/null
export PATH=$HOME/git/fpocket/bin:$PATH
export NPROC=48
cd $HOME/sh
python3 compute_pocket_fv.py ${WORK}/ProtCast/structures
