#!/bin/bash
#SBATCH -J pocket_res
#SBATCH -o pocket_res.%j.out
#SBATCH -e pocket_res.%j.err
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -A MCB22067

# fpocket is CPU-only and runs natively (static binary, no container needed).
export NPROC=48
cd "$WORK/ProtCast" || exit 1
python3 "$HOME/sh/compute_pocket_residues.py"
