"""Generic functional test: does an arbitrary FV (npz pids,S,names) improve GO-MF
over ESM-only? Trains ESM-only vs ESM+FV on the same structure-subset / seeds."""
import os, sys, gc, json, statistics as st
import numpy as np
sys.path.insert(0, os.path.expanduser("~/git/ProtCast/scripts"))
from scan_individual_features import load_flat_embeddings
from protcast.model.multilabel_classifier import MultiLabelClassifier
from protcast.model.stats.utils import calculate_fmax
from protcast.config.model_config import ConfigManager

EMBDIR, FV_NPZ = sys.argv[1], sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
OUT = sys.argv[4]
PREFIX = sys.argv[5] if len(sys.argv) > 5 else "fvtest"

config = ConfigManager.load_config()
emb, goterms, go_ids = load_flat_embeddings(EMBDIR, verbose=True)
z = np.load(FV_NPZ, allow_pickle=True)
Sf = z["S"]; pf = z["pids"]; fv = {p: Sf[i] for i, p in enumerate(pf)}
print(f"FV {os.path.basename(FV_NPZ)}: {len(fv)} proteins, {z['S'].shape[1]} dims {list(z['names'])}")

common = sorted(set(emb) & set(goterms) & set(fv))
print(f"proteins with embedding + GO + FV: {len(common)}")
emb_sub = {p: np.asarray(emb[p], np.float32) for p in common}
combo = {p: np.concatenate([np.asarray(emb[p], np.float32), fv[p]]) for p in common}
go_sub = {p: goterms[p] for p in common}
del emb; gc.collect()
print(f"ESM dim {len(next(iter(emb_sub.values())))}  combined {len(next(iter(combo.values())))}")


def train(embeddings, tag, seed):
    clf = MultiLabelClassifier(verbose=False, protein_embeddings=embeddings,
                               protein_go_terms=go_sub, go_ids=go_ids, config=config,
                               id=f"{PREFIX}_{tag}_s{seed}", random_state=seed, scale_features=True)
    clf.run()
    yp = clf.model.predict(clf.X_val, verbose=0)
    fmax, _ = calculate_fmax(clf.y_val, yp)
    del clf; gc.collect()
    return float(fmax)


rows = []
for seed in SEEDS:
    fb = train(emb_sub, "esmonly", seed)
    ff = train(combo, "esmfv", seed)
    rows.append({"seed": seed, "esm_only": fb, "esm_fv": ff, "delta": ff - fb})
    print(f"seed {seed}: ESM-only {fb:.4f} | ESM+FV {ff:.4f} | delta {ff-fb:+.4f}", flush=True)

deltas = [r["delta"] for r in rows]
mean = st.mean(deltas); sd = st.pstdev(deltas) if len(deltas) > 1 else 0.0
print("\n" + "=" * 50)
print(f"n_seeds={len(deltas)}  proteins={len(common)}")
print(f"mean delta Fmax: {mean:+.4f}  SD {sd:.4f}  all-positive {all(d>0 for d in deltas)}")
json.dump({"proteins": len(common), "rows": rows, "mean_delta": mean, "sd": sd}, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
