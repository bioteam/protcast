"""Build non-contextual feature vectors over the annotated functional residues.

Control for the active-site ESM-C embedding (make_active_site_esm_fv.py): same
proteins, same UniProt ACT_SITE + BINDING positions, but no protein language model.
If the ESM-C site embedding's gain comes only from *which* amino acids sit at the
annotated positions, these features should recover it.

Variants (--variant):
  comp      20-d amino-acid composition of the site residues. Equal to mean-pooling
            one-hot rows over the sites; BLOSUM62 rows are a linear map of it.
  comp_ctx  comp + log(1 + number of site positions) + 20-d composition of the
            flanking residues within +/-W of any site, excluding the sites (41-d).

Positions follow make_active_site_esm_fv.py exactly: the union of ACT_SITE and BINDING
positions, 1-based, keeping 1 <= p <= L. Only proteins in --keep_npz are written, so
the protein set equals that of the ESM-C site embedding (19,294 proteins).
Non-standard residues (X, U, B, Z, O) are not counted.

Output npz: pids, S (M, D), names -- the format fv_go_test.py reads.
"""
import argparse
import json
import math

import numpy as np

from protcast.preprocessing.protcast_dataset import ProtCastDataset

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}


def composition(seq, positions):
    """20-d fraction of standard residues at 1-based positions (zeros if none)."""
    counts = np.zeros(len(AA), dtype=np.float32)
    for p in positions:
        i = AA_IDX.get(seq[p - 1])
        if i is not None:
            counts[i] += 1
    total = counts.sum()
    return counts / total if total else counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--dataset", required=True, help="ProtCastDataset.bin")
    ap.add_argument("--sites", required=True, help="uniprot_sites.json")
    ap.add_argument("--keep_npz", required=True,
                    help="active_site_esm_fv.npz; its pids define the protein set")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--variant", choices=["comp", "comp_ctx"], default="comp")
    ap.add_argument("--window", type=int, default=3, help="flank half-width for comp_ctx")
    args = ap.parse_args()

    sites = json.load(open(args.sites))
    keep = np.load(args.keep_npz, allow_pickle=True)["pids"].tolist()
    ds = ProtCastDataset.load_serialized_file(args.dataset)

    ids, X, n_sites = [], [], []
    missing = 0
    for acc in keep:
        seq = ds.proteins[acc].sequence
        L = len(seq)
        s = sites.get(acc, {})
        pos = sorted({int(p) for p in s.get("act", []) + s.get("binding", [])
                      if 1 <= int(p) <= L})
        if not pos:
            missing += 1
            continue
        vec = [composition(seq, pos)]
        if args.variant == "comp_ctx":
            site_set = set(pos)
            flank = sorted({q for p in pos
                            for q in range(p - args.window, p + args.window + 1)
                            if 1 <= q <= L and q not in site_set})
            vec += [np.array([math.log1p(len(pos))], np.float32), composition(seq, flank)]
        ids.append(acc)
        X.append(np.concatenate(vec))
        n_sites.append(len(pos))

    names = [f"site_{a}" for a in AA]
    if args.variant == "comp_ctx":
        names += ["site_log1p_count"] + [f"flank{args.window}_{a}" for a in AA]
    S = np.asarray(X, dtype=np.float32)
    np.savez_compressed(args.out, pids=np.array(ids), S=S, names=np.array(names))
    print(f"{args.variant}: wrote {len(ids)} proteins x {S.shape[1]} dims -> {args.out}")
    print(f"proteins in keep list without a usable site: {missing}")
    print(f"site positions per protein: median {int(np.median(n_sites))}, "
          f"max {max(n_sites)}")


if __name__ == "__main__":
    main()
