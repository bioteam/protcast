"""Per-GO-term AP AND per-term Fmax (max-F1 over thresholds) for an arbitrary FV:
ESM-only vs ESM+FV. Generalized from eval_pocket_aa_perterm.py.

Inference only -- reuses saved {prefix}_{tag}_s{seed}_multilabel.keras checkpoints
from a fv_go_test.py run. For each seed, reconstructs the exact val split
(train_test_split(random_state=seed)) over the SAME `common` protein set both arms
trained on, refits the train-fold StandardScaler, and computes per-term AP and Fmax
for base and FV.

Both arms trained on `common` (proteins with embedding + GO + FV), so base/FV val
sets are identical per seed -> fully paired per-term comparison.

Output: long-format CSV (one row per term x seed):
  go_id, seed, support, ap_base, ap_fv, fmax_base, fmax_fv

Caveat: on-disk .keras is best-val-loss checkpoint, not best-Fmax restored weights,
and per-term Fmax uses the per-term-optimal threshold on the val fold (optimistic but
symmetric across arms) -- relative base-vs-FV comparison is fair.
"""
import os
import sys
import argparse
import gc
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, precision_recall_curve
import keras

sys.path.insert(0, os.path.expanduser("~/git/ProtCast/scripts"))
from scan_individual_features import load_flat_embeddings
from protcast.config.model_config import ConfigManager


def build_Xy(emb_dict, go_terms, cols):
    g2i = {g: i for i, g in enumerate(cols)}
    pids = sorted(set(emb_dict.keys()) & set(go_terms.keys()))
    X = np.vstack([np.asarray(emb_dict[p], dtype=np.float32) for p in pids])
    y = np.zeros((len(pids), len(cols)), dtype=np.float32)
    for r, p in enumerate(pids):
        for g in go_terms[p]:
            if g in g2i:
                y[r, g2i[g]] = 1.0
    return X, y


def val_predict(X, y, seed, val_split, model_path):
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=val_split, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    Xval_s = scaler.transform(Xval).astype(np.float32)
    model = keras.models.load_model(model_path, compile=False)
    pred = model.predict(Xval_s, batch_size=512, verbose=0)
    del model, Xtr, Xval, Xval_s
    gc.collect()
    return yval, pred


def term_fmax(ytrue, score):
    """Max F1 over all thresholds for one term."""
    p, r, _ = precision_recall_curve(ytrue, score)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    return float(f1.max())


def per_term_metrics(yval, pred):
    out = {}
    for j in range(yval.shape[1]):
        sup = int(yval[:, j].sum())
        if sup > 0:
            out[j] = (sup,
                      float(average_precision_score(yval[:, j], pred[:, j])),
                      term_fmax(yval[:, j], pred[:, j]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embdir", required=True)
    ap.add_argument("--fv_npz", required=True)
    ap.add_argument("--modeldir", required=True)
    ap.add_argument("--model_prefix", default="fvtest")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = ConfigManager.load_config()
    val_split = 0.2
    for k, v in config.items():
        if k.lower() == "validation_split":
            val_split = float(v)
    print(f"validation_split = {val_split}", flush=True)

    emb, goterms, go_ids = load_flat_embeddings(args.embdir, verbose=True)
    cols = sorted(set(go_ids))
    z = np.load(args.fv_npz, allow_pickle=True)
    Sf, pf = z["S"], z["pids"]
    fv = {p: Sf[i] for i, p in enumerate(pf)}

    common = sorted(set(emb) & set(goterms) & set(fv))
    print(f"common proteins (emb & GO & FV): {len(common)}", flush=True)
    emb_sub = {p: np.asarray(emb[p], np.float32) for p in common}
    combo = {p: np.concatenate([np.asarray(emb[p], np.float32), fv[p]]) for p in common}
    go_sub = {p: goterms[p] for p in common}
    del emb
    gc.collect()

    Xb, yb = build_Xy(emb_sub, go_sub, cols)
    Xf, yf = build_Xy(combo, go_sub, cols)
    print(f"ESM dim {Xb.shape[1]}  combined {Xf.shape[1]}", flush=True)

    rows = []
    for seed in args.seeds:
        bm = os.path.join(args.modeldir, f"{args.model_prefix}_esmonly_s{seed}_multilabel.keras")
        fm = os.path.join(args.modeldir, f"{args.model_prefix}_esmfv_s{seed}_multilabel.keras")
        if not (os.path.exists(bm) and os.path.exists(fm)):
            print(f"  seed {seed}: MISSING checkpoints, skipping", flush=True)
            continue
        yvb, pb = val_predict(Xb, yb, seed, val_split, bm)
        mb = per_term_metrics(yvb, pb)
        yvf, pf2 = val_predict(Xf, yf, seed, val_split, fm)
        mf = per_term_metrics(yvf, pf2)
        common_j = set(mb) & set(mf)
        wsup = sum(mb[j][0] for j in common_j)
        wd_ap = sum(mb[j][0] * (mf[j][1] - mb[j][1]) for j in common_j) / wsup
        wd_fmax = sum(mb[j][0] * (mf[j][2] - mb[j][2]) for j in common_j) / wsup
        print(f"  seed {seed}: terms={len(common_j)}  support-wtd dAP={wd_ap:+.4f}  dFmax={wd_fmax:+.4f}", flush=True)
        for j in common_j:
            rows.append({"go_id": cols[j], "seed": seed, "support": mb[j][0],
                         "ap_base": mb[j][1], "ap_fv": mf[j][1],
                         "fmax_base": mb[j][2], "fmax_fv": mf[j][2]})
        del yvb, pb, yvf, pf2
        gc.collect()

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["go_id", "seed", "support",
                                          "ap_base", "ap_fv", "fmax_base", "fmax_fv"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
