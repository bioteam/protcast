#!/bin/bash
#SBATCH --job-name run_make_esm_embeddings
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aakpan@bioteam.net
#SBATCH -o run_make_esm_embeddings.out
#SBATCH -e run_make_esm_embeddings.err
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 24:00:00

CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
WORK_DIR=${WORK}/ProtCast/ProtCastDataset/01-23-2026/
DATASET=${WORK_DIR}/ProtCastDataset.bin
MIN_SEQ=200

# Directory for Hugging Face
mkdir -p ${WORK}/HF
export HF_HOME=${WORK}/HF

source ${HOME}/.bash_profile
# Only use modules from the container
unset PYTHONPATH
module load tacc-apptainer

cd ${HOME}/git/ProtCast/

# Generate both pooling variants. `mean` is the defensible primary baseline;
# `mean_max_std` is the sensitivity variant. `mean` runs first so the primary
# baseline is available even if the job times out.
#
# Special tokens (BOS/EOS) are now stripped by default, so:
#   * the "-stripped" suffix documents the recipe, and
#   * writing to fresh directories avoids the skip-if-exists guard in
#     make_esm_embeddings.py, which would otherwise leave the OLD, unstripped
#     embeddings untouched and produce nothing new.
# --force is intentionally omitted so a re-run after a timeout resumes by
# skipping already-completed GO terms.
for POOL in mean mean_max_std
do
	for NUM in {0..10}
	do
		singularity exec --nv $CONTAINER \
			python3 scripts/make_esm_embeddings.py \
			-v \
			--minimum_seqs $MIN_SEQ \
			-p $DATASET \
			-g $WORK_DIR/mf_go_terms-level-${NUM}.tsv \
			-o $WORK_DIR/mf_go_terms-level-${NUM}-${POOL}-stripped \
			--pooling ${POOL}
	done
done
