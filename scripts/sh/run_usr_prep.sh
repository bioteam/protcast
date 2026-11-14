#!/bin/bash
#SBATCH -J usr_prep
#SBATCH -p small
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:40:00
#SBATCH -o usr_prep.out
#SBATCH -e usr_prep.err
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
STRUCTDIR=${WORK}/ProtCast/structures
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast
if [ ! -s $STRUCTDIR/usr_fv.npz ]; then
  echo "### featurize USR ###"
  singularity exec $CONTAINER python3 $HOME/sh/compute_usr_fv.py $STRUCTDIR
else
  echo "### usr_fv.npz exists, skipping featurize ###"
fi
echo "### orthogonality ESM -> USR ###"
singularity exec $CONTAINER python3 $HOME/sh/ortho_npz.py $STRUCTDIR/sample_emb.npz $STRUCTDIR/usr_fv.npz $STRUCTDIR/usr_orthogonality_R2.csv
