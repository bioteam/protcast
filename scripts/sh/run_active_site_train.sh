#!/bin/bash
#SBATCH -J active_site_train
#SBATCH -p rtx
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 06:00:00
#SBATCH -o active_site_train.out
#SBATCH -e active_site_train.err
#SBATCH -A MCB22067
#SBATCH --mail-type=ALL
#SBATCH --mail-user=briano@bioteam.net
# Exp #3: ESM-C vs ESM-C + active-site AA-composition FV (active_site_fv.npz).
# Repeated against the NEW strip-fixed mean_max_std embeddings, GO levels 5-8, 5 seeds.
# Per-level result JSONs (active_site_go_result_L<level>.json) leave the old
# pre-strip aggregate (active_site_go_result.json) intact for comparison.
CONTAINER=${WORK}/tensorflow_2.17.0-gpu.sif
DATA=${WORK}/ProtCast/ProtCastDataset/01-23-2026
FV=${WORK}/ProtCast/active_site_fv.npz
SEEDS=${SEEDS:-1,42,555,5555,5}
LEVELS=${LEVELS:-"5 6 7 8"}
export PYTHONPATH=$HOME/.local/lib/python3.11/site-packages
module load tacc-apptainer
cd $HOME/git/ProtCast
for L in ${LEVELS}; do
    EMBDIR=${DATA}/mf_go_terms-level-${L}-mean_max_std
    OUT=${WORK}/ProtCast/active_site_go_result_L${L}.json
    echo "==================================================================="
    echo "Exp3 active_site_fv | level ${L} | seeds ${SEEDS}"
    echo "  EMBDIR=${EMBDIR}"
    echo "==================================================================="
    singularity exec --nv $CONTAINER python3 $HOME/sh/fv_go_test.py \
        "$EMBDIR" "$FV" "${SEEDS}" "$OUT" active_site 2>&1 | tee ~/sh/active_site_L${L}.log
done
