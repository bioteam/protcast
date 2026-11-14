#!/bin/bash
#SBATCH -J pocket_aa_perterm
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 02:00:00
#SBATCH -o pocket_aa_perterm.out
#SBATCH -e pocket_aa_perterm.err
#SBATCH -A MCB22067
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
EMBDIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026/mf_go_terms-level-5-mean_max_std
FV=${WORK}/ProtCast/pocket_aa_fv.npz
OBO=${WORK}/ProtCast/GO/2025-10-10/go.obo
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast
# All 12 seeds; eval skips any whose checkpoints aren't present yet.
singularity exec $CONTAINER python3 $HOME/sh/eval_pocket_aa_perterm.py \
  --embdir $EMBDIR --fv_npz $FV --modeldir $HOME/git/ProtCast \
  --seeds 1 42 555 5555 5 22 7 13 99 2024 314 271 \
  --out ${WORK}/ProtCast/pocket_aa_perterm.csv
singularity exec $CONTAINER python3 $HOME/sh/map_catalytic.py \
  --perterm ${WORK}/ProtCast/pocket_aa_perterm.csv --obo $OBO \
  --out ${WORK}/ProtCast/pocket_aa_catalytic.json
