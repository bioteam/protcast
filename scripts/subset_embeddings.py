"""Copy a level's per-GO-term embedding pickles, keeping only the proteins in a .npz.

Used to shrink the mean_max_std embedding directories to the annotated-site subset
(the pids of active_site_esm_fv.npz) before moving them off Frontera. Every GO_*.pkl
file is written, even when no protein survives, so that load_flat_embeddings()
returns the same ordered go_ids list as the full directory.

    python3 subset_embeddings.py IN_DIR KEEP_NPZ OUT_DIR
"""
import os
import pickle
import sys

import numpy as np


def main():
    in_dir, keep_npz, out_dir = sys.argv[1:4]
    keep = set(np.load(keep_npz, allow_pickle=True)["pids"].tolist())
    os.makedirs(out_dir, exist_ok=True)
    proteins, files = set(), 0
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".pkl"):
            continue
        with open(os.path.join(in_dir, name), "rb") as f:
            emb = pickle.load(f)
        sub = {p: v for p, v in emb.items() if p in keep}
        with open(os.path.join(out_dir, name), "wb") as f:
            pickle.dump(sub, f, protocol=4)
        proteins.update(sub)
        files += 1
    print("{}: {} GO-term files, {} of {} kept proteins present".format(
        out_dir, files, len(proteins), len(keep)))


if __name__ == "__main__":
    main()
