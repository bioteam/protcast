#!/bin/bash
#SBATCH -J as_aa_perterm
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o as_aa_perterm.out
#SBATCH -e as_aa_perterm.err
#SBATCH -A MCB22067
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
EMBDIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026/mf_go_terms-level-5-mean_max_std
FV=${WORK}/ProtCast/active_site_aa_only_fv.npz
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast
singularity exec $CONTAINER python3 $HOME/sh/eval_fv_perterm.py \
  --embdir $EMBDIR --fv_npz $FV --modeldir $HOME/git/ProtCast \
  --model_prefix as_aa \
  --seeds 1 42 555 5555 5 22 7 13 99 2024 314 271 \
  --out ${WORK}/ProtCast/as_aa_perterm.csv
