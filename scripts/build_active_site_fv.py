#!/usr/bin/env python3
"""Build a single-pocket amino-acid FV anchored on the ENZYME ACTIVE SITE.

For each protein, pick the fpocket cavity that best matches the UniProt
ACT_SITE/BINDING residues (most catalytic residues inside it), and emit that one
pocket's 20-dim AA fraction. This isolates active-site chemistry, removing the
top-3 dilution and fpocket's weak score-ranking that the geometry/pocket_aa FVs
suffered from.

Pocket selection per protein (lexicographic):
  max #ACT_SITE residues in pocket, then #BINDING residues, then fpocket score,
  then #lining residues. If no annotated site lands in any pocket (or no
  annotation), fall back to the largest cleft (most lining residues) and set
  has_site=0.

Output: $WORK/ProtCast/active_site_fv.npz
  pids  -> accessions
  S     -> (M, 22): 20 AA fraction of chosen pocket + site_overlap + has_site
  names
Also prints coverage: how often a real active-site pocket was found (= fpocket
recall of the annotated site on AF structures).
"""
import os
import json
import numpy as np

WORK = os.environ["WORK"]
JSONL = os.path.join(WORK, "ProtCast", "pocket_residues.jsonl")
SITES = os.path.join(WORK, "ProtCast", "uniprot_sites.json")
OUT = os.path.join(WORK, "ProtCast", "active_site_fv.npz")

AA_ORDER = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA_IDX = {a: i for i, a in enumerate(AA_ORDER)}


def accession(name):
    parts = name.split("-")
    return parts[1] if len(parts) >= 2 and parts[0] == "AF" else name


def main():
    sites = json.load(open(SITES))

    # group pockets by accession
    prot = {}  # acc -> list of pockets {resseq:set(int), aa:dict, score, n_res}
    with open(JSONL) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "aa" not in r or not r.get("n_res"):
                continue
            acc = accession(r["id"])
            resseq = set()
            for c, s, rn in r["residues"]:
                try:
                    resseq.add(int(s))
                except ValueError:
                    pass
            prot.setdefault(acc, []).append(
                {"resseq": resseq, "aa": r["aa"], "score": r.get("score") or 0.0,
                 "n_res": r["n_res"]})

    # also need every protein that exists at all (incl. no-pocket) -> from sites keys
    all_accs = sorted(set(sites.keys()) | set(prot.keys()))

    ids, X = [], []
    n_found = n_fallback = n_nopocket = n_annot = 0
    for acc in all_accs:
        pockets = prot.get(acc, [])
        vec = np.zeros(22, dtype=np.float32)
        if not pockets:
            n_nopocket += 1
            ids.append(acc); X.append(vec); continue
        s = sites.get(acc, {"act": [], "binding": []})
        act, bind = set(s["act"]), set(s["binding"])
        has_annot = bool(act or bind)
        if has_annot:
            n_annot += 1

        def key(p):
            return (len(p["resseq"] & act), len(p["resseq"] & bind),
                    p["score"], p["n_res"])

        best = max(pockets, key=key)
        a_ov = len(best["resseq"] & act)
        b_ov = len(best["resseq"] & bind)
        overlap = a_ov + b_ov
        if overlap > 0:
            n_found += 1
            has_site = 1.0
        else:
            # fallback: largest cleft
            best = max(pockets, key=lambda p: p["n_res"])
            n_fallback += 1
            has_site = 0.0

        total = sum(best["aa"].values())
        if total:
            for rn, cnt in best["aa"].items():
                j = AA_IDX.get(rn)
                if j is not None:
                    vec[j] = cnt / total
        vec[20] = overlap
        vec[21] = has_site
        ids.append(acc); X.append(vec)

    names = [f"as_{a}" for a in AA_ORDER] + ["site_overlap", "has_site"]
    np.savez_compressed(OUT, pids=np.array(ids),
                        S=np.asarray(X, dtype=np.float32), names=np.array(names))
    print(f"{len(ids)} proteins -> {OUT}")
    print(f"  with UniProt annotation: {n_annot}")
    print(f"  active-site pocket FOUND (annotated residue inside a pocket): {n_found}")
    print(f"  fallback to largest cleft (no annot or site not in any pocket): {n_fallback}")
    print(f"  no pocket at all: {n_nopocket}")


if __name__ == "__main__":
    main()
