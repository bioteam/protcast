"""Functional test: does the 4-dim shape FV improve GO-MF prediction over ESM-only?

Trains ESM-only baseline and ESM+shape on the SAME protein subset (those with a
structure) and identical seeds/splits, via MultiLabelClassifier(scale_features=True)
-- so the shape features are z-scored on the train fold alongside the ESM block.
Reports Fmax per seed and the delta, to compare against the composition-FV deltas.
"""
import os, sys, gc, json, statistics as st
import numpy as np
sys.path.insert(0, os.path.expanduser("~/git/ProtCast/scripts"))
from scan_individual_features import load_flat_embeddings
from protcast.model.multilabel_classifier import MultiLabelClassifier
from protcast.model.stats.utils import calculate_fmax
from protcast.config.model_config import ConfigManager

EMBDIR = sys.argv[1]
STRUCTDIR = sys.argv[2]
SEEDS = [int(x) for x in sys.argv[3].split(",")]
OUT = sys.argv[4]

config = ConfigManager.load_config()
emb, goterms, go_ids = load_flat_embeddings(EMBDIR, verbose=True)

z = np.load(os.path.join(STRUCTDIR, "shape_fv.npz"), allow_pickle=True)
shape = {p: z["S"][i] for i, p in enumerate(z["pids"])}
print(f"shape FVs: {len(shape)}  features: {list(z['names'])}")

common = sorted(set(emb) & set(goterms) & set(shape))
print(f"proteins with embedding + GO + structure: {len(common)}")

emb_sub = {p: np.asarray(emb[p], np.float32) for p in common}
combo = {p: np.concatenate([np.asarray(emb[p], np.float32), shape[p]]) for p in common}
go_sub = {p: goterms[p] for p in common}
del emb; gc.collect()
print(f"ESM dim {len(next(iter(emb_sub.values())))}  combined dim {len(next(iter(combo.values())))}")


def train(embeddings, tag, seed):
    clf = MultiLabelClassifier(
        verbose=False, protein_embeddings=embeddings, protein_go_terms=go_sub,
        go_ids=go_ids, config=config, id=f"shapetest_{tag}_s{seed}",
        random_state=seed, scale_features=True,
    )
    clf.run()
    yp = clf.model.predict(clf.X_val, verbose=0)
    fmax, _ = calculate_fmax(clf.y_val, yp)
    del clf; gc.collect()
    return float(fmax)


rows = []
for seed in SEEDS:
    fb = train(emb_sub, "esmonly", seed)
    fs = train(combo, "esmshape", seed)
    d = fs - fb
    print(f"seed {seed}: ESM-only {fb:.4f} | ESM+shape {fs:.4f} | delta {d:+.4f}", flush=True)
    rows.append({"seed": seed, "esm_only": fb, "esm_shape": fs, "delta": d})

deltas = [r["delta"] for r in rows]
n = len(deltas)
mean = st.mean(deltas)
sd = st.pstdev(deltas) if n > 1 else 0.0
print("\n" + "=" * 50)
print(f"n_seeds={n}  proteins={len(common)}")
print(f"mean delta Fmax (ESM+shape - ESM-only): {mean:+.4f}")
print(f"per-seed SD: {sd:.4f}   all-positive: {all(d>0 for d in deltas)}")
json.dump({"proteins": len(common), "rows": rows, "mean_delta": mean, "sd": sd},
          open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
