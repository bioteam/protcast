"""Build an active-site-LOCAL ESM-C FV: for each protein with UniProt ACT_SITE/BINDING
annotation, run ESM-C, take the per-residue embeddings at ONLY the annotated site
positions, and mean-pool them -> one D-dim vector per protein.

Rationale: ProtCast's whole-protein embedding is mean/max/std-pooled over the entire
sequence, so the handful of catalytic residues are drowned (~1/L weight). Pooling ESM-C
over just the site residues amplifies that local context ~L-fold. This is the only
active-site feature that could carry info the global pool loses (composition already
failed). Only annotated proteins are emitted, so downstream common-set = annotated
subset (no has_site indicator confound; every protein has a site).

Output npz: pids, S (M, D), names.
"""
import os, json, argparse, random
import numpy as np
import torch
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein

from protcast.preprocessing.protcast_dataset import ProtCastDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--dataset", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--model_type", default="esmc_600m")
    ap.add_argument("--residues", default="act_binding", choices=["act", "act_binding"])
    ap.add_argument("--random", action="store_true",
                    help="CONTROL: replace each protein's site positions with the SAME "
                         "NUMBER of random positions (deterministic per accession).")
    ap.add_argument("--random_seed", default="ctrl", help="seed tag for --random")
    ap.add_argument("--limit", type=int, default=0, help="process only first N (dry run)")
    args = ap.parse_args()

    sites = json.load(open(args.sites))
    ds = ProtCastDataset.load_serialized_file(args.dataset)
    ds_pids = set(ds.proteins.keys())
    print(f"dataset proteins: {len(ds_pids)}; sites accs: {len(sites)}; "
          f"overlap: {len(ds_pids & set(sites))}", flush=True)

    # accessions with >=1 annotated site position
    def positions(acc):
        s = sites.get(acc, {})
        pos = set(s.get("act", []))
        if args.residues == "act_binding":
            pos |= set(s.get("binding", []))
        return sorted(int(p) for p in pos)

    todo = [a for a in sorted(sites) if a in ds_pids and positions(a)]
    if args.limit:
        todo = todo[:args.limit]
    print(f"proteins with >=1 {args.residues} position AND sequence: {len(todo)}", flush=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}", flush=True)
    model = ESMC.from_pretrained(args.model_type, device=device)
    model.eval()

    ids, X = [], []
    n_off = {}
    skipped = 0
    for i, acc in enumerate(todo):
        seq = ds.proteins[acc].sequence
        L = len(seq)
        pos = [p for p in positions(acc) if 1 <= p <= L]
        if not pos:
            skipped += 1
            continue
        if args.random:
            # same COUNT of positions, drawn randomly from the sequence,
            # deterministic per accession so the control is reproducible
            k = min(len(pos), L)
            rng = random.Random(f"{args.random_seed}:{acc}")
            pos = sorted(rng.sample(range(1, L + 1), k))
        try:
            with torch.no_grad():
                pt = model.encode(ESMProtein(sequence=seq))
                out = model.forward(sequence_tokens=pt.sequence.unsqueeze(0))
                emb = out.embeddings.squeeze(0).to(dtype=torch.float32)  # (rows, D)
            rows = emb.shape[0]
            offset = rows - L                    # #special tokens (expect BOS+EOS=2)
            n_off[offset] = n_off.get(offset, 0) + 1
            has_bos = rows > L                    # leading BOS present?
            idx = [(p if has_bos else p - 1) for p in pos]
            idx = [j for j in idx if 0 <= j < rows]
            if not idx:
                skipped += 1
                continue
            vec = emb[idx].mean(dim=0).cpu().numpy()
            if not ids:  # first kept protein: sanity-print indexing
                print(f"  [sanity] {acc} L={L} rows={rows} offset={offset} "
                      f"has_bos={has_bos} pos={pos[:5]} idx={idx[:5]} D={vec.shape[0]}",
                      flush=True)
            ids.append(acc); X.append(vec)
        except Exception as e:
            print(f"  ERR {acc}: {e}", flush=True)
            skipped += 1
            continue
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(todo)} done ({len(ids)} kept)", flush=True)

    D = len(X[0]) if X else 0
    names = [f"asemb_{k}" for k in range(D)]
    np.savez_compressed(args.out, pids=np.array(ids),
                        S=np.asarray(X, dtype=np.float32), names=np.array(names))
    print(f"wrote {len(ids)} proteins x {D} dims -> {args.out}", flush=True)
    print(f"skipped: {skipped}; special-token offsets seen: {n_off}", flush=True)


if __name__ == "__main__":
    main()
