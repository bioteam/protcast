#!/bin/bash
# Site-residue controls for the active-site ESM-C embedding (manuscript Table 5).
# Runs ESM-C alone vs ESM-C + each feature file on the 19,294 proteins with annotated
# active-site or binding positions, GO levels 5-8, seeds 1, 42, 555, 5555, 5:
#   site_esm      active_site_esm_fv.npz (1152-d, ESM-C pooled over the sites)
#   site_comp     site_comp_fv.npz       (20-d amino-acid composition of the site residues)
#   site_comp_ctx site_comp_ctx_fv.npz   (41-d: composition + log site count + +/-3 flank composition)
#
# DATA_DIR must contain:
#   emb/emb-L{5..8}   per-level mean_max_std embeddings restricted to the annotated proteins:
#                       python3 scripts/subset_embeddings.py <level-L embedding dir> active_site_esm_fv.npz emb/emb-L<L>
#   active_site_esm_fv.npz          from scripts/make_active_site_esm_fv.py
#   site_comp_fv.npz, site_comp_ctx_fv.npz
#                       python3 scripts/make_site_composition_fv.py -p ProtCastDataset.bin --sites uniprot_sites.json \
#                         --keep_npz active_site_esm_fv.npz -o site_comp_fv.npz --variant comp
#                       (and --variant comp_ctx -o site_comp_ctx_fv.npz)
#   config.json       training settings read by fv_go_test.py (EPOCHS, BATCH_SIZE, PATIENCE, ...)
# Results go to DATA_DIR/out/<arm>_L<level>.json and logs to DATA_DIR/logs/. Finished runs are skipped.
#
# Usage: DATA_DIR=/path/to/site-control PYTHON=/path/to/python scripts/sh/run_site_control.sh
# The manuscript run used TensorFlow 2.17 on a CPU.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
DATA_DIR=${DATA_DIR:-$PWD}
PYTHON=${PYTHON:-python3}
FVTEST=$REPO/scripts/fv_go_test.py
SEEDS=1,42,555,5555,5
export PYTHONPATH=$REPO/scripts:$REPO${PYTHONPATH:+:$PYTHONPATH} TF_CPP_MIN_LOG_LEVEL=2

cd "$DATA_DIR" || exit 1
mkdir -p out logs
for L in 8 7 6 5; do
  for FV in site_esm:active_site_esm_fv.npz site_comp:site_comp_fv.npz site_comp_ctx:site_comp_ctx_fv.npz; do
    TAG=${FV%%:*}; NPZ=${FV#*:}
    [ -s out/${TAG}_L${L}.json ] && { echo "skip ${TAG} L${L}"; continue; }
    echo "=== $(date '+%F %T') ${TAG} level ${L}"
    "$PYTHON" "$FVTEST" emb/emb-L${L} "$NPZ" $SEEDS out/${TAG}_L${L}.json ${TAG} \
      > logs/${TAG}_L${L}.log 2>&1
    grep -E '^seed|mean delta' logs/${TAG}_L${L}.log
  done
done
echo "=== $(date '+%F %T') done"
