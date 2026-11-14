#!/usr/bin/env python3
"""Aggregate pocket_residues.jsonl into a per-protein amino-acid-composition FV.

Mirrors the earlier pocket-geometry FV (top-N pockets by fpocket Score) but with
pocket *chemistry*: each of the top-N pockets becomes a 20-dim amino-acid
FRACTION vector (fraction of that pocket's lining residues that are each AA),
concatenated, optionally with n_pockets appended. Proteins with fewer than N
pockets are zero-padded; proteins with none are all-zero.

Output: $WORK/ProtCast/pocket_aa_fv.npz with
  pids   -> (M,) UniProt accessions (parsed from AF-{acc}-F1-model_v6)
  S      -> (M, 20*N [+1]) float32
  names  -> (D,) feature names
pids/S match the pocket_fv.npz / usr_fv.npz key convention that fv_go_test.py
reads (z["pids"], z["S"]), so the existing harness consumes it directly.
"""
import os
import json
from collections import defaultdict

import numpy as np

WORK = os.environ["WORK"]
JSONL = os.path.join(WORK, "ProtCast", "pocket_residues.jsonl")
OUT = os.path.join(WORK, "ProtCast", "pocket_aa_fv.npz")
NPOCKETS = int(os.environ.get("NPOCKETS", "3"))
ADD_NPOCKETS = os.environ.get("ADD_NPOCKETS", "1") != "0"

AA_ORDER = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_IDX = {a: i for i, a in enumerate(AA_ORDER)}


def accession(name):
    # AF-{acc}-F1-model_v6 -> {acc}; fall back to the raw id
    parts = name.split("-")
    return parts[1] if len(parts) >= 2 and parts[0] == "AF" else name


def main():
    pockets = defaultdict(list)   # id -> list of (score, aa_dict)
    n_pockets = defaultdict(int)  # id -> count of real pockets
    seen = set()
    with open(JSONL) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            pid = row.get("id")
            if pid is None:
                continue
            seen.add(pid)
            if "aa" in row and row.get("n_res"):
                pockets[pid].append((row.get("score"), row["aa"]))
                n_pockets[pid] += 1

    ids, X = [], []
    dim = 20 * NPOCKETS + (1 if ADD_NPOCKETS else 0)
    for pid in sorted(seen):
        # rank pockets by score (missing score sorts last)
        ranked = sorted(pockets.get(pid, []),
                        key=lambda t: (t[0] is None, -(t[0] or 0.0)))
        vec = np.zeros(dim, dtype=np.float32)
        for k, (_, aa) in enumerate(ranked[:NPOCKETS]):
            total = sum(aa.values())
            if total:
                base = 20 * k
                for res, cnt in aa.items():
                    j = AA_IDX.get(res)
                    if j is not None:
                        vec[base + j] = cnt / total
        if ADD_NPOCKETS:
            vec[-1] = n_pockets.get(pid, 0)
        ids.append(accession(pid))
        X.append(vec)

    names = [f"p{k+1}_{a}" for k in range(NPOCKETS) for a in AA_ORDER]
    if ADD_NPOCKETS:
        names.append("n_pockets")

    np.savez_compressed(OUT,
                        pids=np.array(ids),
                        S=np.asarray(X, dtype=np.float32),
                        names=np.array(names))
    print(f"{len(ids)} proteins, X shape ({len(ids)}, {dim}) -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
