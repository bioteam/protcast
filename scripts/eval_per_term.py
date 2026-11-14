"""Per-GO-term Average Precision: ESM-only baseline vs ESM+FV.

Inference only -- reuses the saved .keras models from the candidate scan.
Reconstructs each model's exact val split (train_test_split(random_state=seed))
and refits the train-fold StandardScaler deterministically. No retraining.

Caveat: the on-disk .keras is the best-val-loss checkpoint (ModelCheckpoint),
not the best-Fmax restored weights, so recomputed aggregate Fmax will not
exactly match the scan JSON. Relative baseline-vs-FV comparison is still fair.
The FV path drops ~494 proteins, so baseline and FV val sets overlap but are
not perfectly paired; averaging over seeds mitigates this.
"""
import os, sys, argparse, gc, csv, json
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import keras

sys.path.insert(0, os.path.expanduser("~/git/ProtCast/scripts"))
from scan_individual_features import load_flat_embeddings, build_combined_embeddings
from protcast.preprocessing.protcast_dataset import ProtCastDataset
from protcast.config.model_config import ConfigManager


def build_Xy(emb_dict, go_terms, go_ids):
    cols = sorted(set(go_ids))
    g2i = {g: i for i, g in enumerate(cols)}
    pids = sorted(set(emb_dict.keys()) & set(go_terms.keys()))
    X = np.vstack([np.asarray(emb_dict[p], dtype=np.float32) for p in pids])
    y = np.zeros((len(pids), len(cols)), dtype=np.float32)
    for r, p in enumerate(pids):
        for g in go_terms[p]:
            if g in g2i:
                y[r, g2i[g]] = 1.0
    return X, y, cols


def val_predict(X, y, seed, val_split, model_path):
    Xtr, Xval, ytr, yval = train_test_split(
        X, y, test_size=val_split, random_state=seed
    )
    scaler = StandardScaler().fit(Xtr)
    Xval_s = scaler.transform(Xval).astype(np.float32)
    model = keras.models.load_model(model_path, compile=False)
    pred = model.predict(Xval_s, batch_size=512, verbose=0)
    del model, Xtr, Xval, Xval_s
    gc.collect()
    return yval, pred


def per_term_ap(yval, pred):
    """Return dict term_idx -> (support, AP) for terms with >=1 positive."""
    out = {}
    for j in range(yval.shape[1]):
        sup = int(yval[:, j].sum())
        if sup > 0:
            out[j] = (sup, float(average_precision_score(yval[:, j], pred[:, j])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pooling", required=True, choices=["mean", "mean_max_std"])
    ap.add_argument("--level", type=int, required=True)
    ap.add_argument("--algo", default="GAAC")
    ap.add_argument("--datadir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    config = ConfigManager.load_config()
    val_split = 0.2
    for k, v in config.items():
        if k.lower() == "validation_split":
            val_split = float(v)
    print(f"validation_split = {val_split}")

    suf = "-mean_max_std" if args.pooling == "mean_max_std" else ""
    embdir = os.path.join(args.datadir, f"mf_go_terms-level-{args.level}{suf}")
    name = os.path.basename(embdir)
    binpath = os.path.join(args.datadir, "ProtCastDataset.bin")

    print("Loading dataset + embeddings ...")
    dataset = ProtCastDataset.load_serialized_file(binpath)
    emb, goterms, go_ids = load_flat_embeddings(embdir, verbose=True)
    cols = sorted(set(go_ids))

    print(f"Building {args.algo} combined embeddings ...")
    combined, emb_dim, fv_dim = build_combined_embeddings(emb, dataset, args.algo, verbose=True)
    fg = {p: goterms[p] for p in combined if p in goterms}

    # accumulators across seeds: term_idx -> lists
    acc = defaultdict(lambda: {"sup": [], "ap_b": [], "ap_f": []})
    agg_b, agg_f = [], []  # (support, ap) flattened per seed for micro summary

    for seed in args.seeds:
        mdir = os.path.join(args.workdir, f"feature_scan_candidates-{args.pooling}-level-{args.level}-seed-{seed}")
        base_model = os.path.join(mdir, f"{name}_scan_esm_only_multilabel.keras")
        fv_model = os.path.join(mdir, f"{name}_scan_{args.algo}_multilabel.keras")
        if not (os.path.exists(base_model) and os.path.exists(fv_model)):
            print(f"  seed {seed}: MISSING models, skipping ({base_model})")
            continue
        print(f"\n=== seed {seed} ===")

        Xb, yb, _ = build_Xy(emb, goterms, go_ids)
        yvb, pb = val_predict(Xb, yb, seed, val_split, base_model)
        ap_b = per_term_ap(yvb, pb)
        del Xb, yb, yvb, pb; gc.collect()

        Xf, yf, _ = build_Xy(combined, fg, go_ids)
        yvf, pf = val_predict(Xf, yf, seed, val_split, fv_model)
        ap_f = per_term_ap(yvf, pf)
        del Xf, yf, yvf, pf; gc.collect()

        common = set(ap_b) & set(ap_f)
        wb = sum(ap_b[j][0] * ap_b[j][1] for j in common)
        wf = sum(ap_f[j][0] * ap_f[j][1] for j in common)
        wsup = sum(ap_b[j][0] for j in common)
        print(f"  terms scored: {len(common)} | support-weighted AP  base={wb/wsup:.4f}  fv={wf/wsup:.4f}  delta={(wf-wb)/wsup:+.4f}")
        for j in common:
            acc[j]["sup"].append(ap_b[j][0])
            acc[j]["ap_b"].append(ap_b[j][1])
            acc[j]["ap_f"].append(ap_f[j][1])

    # write per-term CSV
    rows = []
    for j, d in acc.items():
        n = len(d["ap_b"])
        if n == 0:
            continue
        mb = float(np.mean(d["ap_b"])); mf = float(np.mean(d["ap_f"]))
        deltas = np.array(d["ap_f"]) - np.array(d["ap_b"])
        rows.append({
            "go_id": cols[j],
            "n_seeds": n,
            "support_mean": float(np.mean(d["sup"])),
            "ap_base": mb,
            "ap_fv": mf,
            "delta": mf - mb,
            "delta_std": float(np.std(deltas)),
            "pos_seeds": int((deltas > 0).sum()),
        })
    rows.sort(key=lambda r: r["delta"], reverse=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary
    deltas = np.array([r["delta"] for r in rows])
    sup = np.array([r["support_mean"] for r in rows])
    print("\n" + "=" * 60)
    print(f"POOLING={args.pooling} LEVEL={args.level} ALGO={args.algo}")
    print(f"terms: {len(rows)}")
    print(f"support-weighted mean delta AP: {np.average(deltas, weights=sup):+.4f}")
    print(f"unweighted mean delta AP:       {deltas.mean():+.4f}")
    print(f"terms improved (delta>0):  {(deltas>0).sum()}/{len(rows)}")
    print(f"terms improved >+0.01:     {(deltas>0.01).sum()}")
    print(f"terms consistently up (pos_seeds==n_seeds): {sum(1 for r in rows if r['pos_seeds']==r['n_seeds'] and r['n_seeds']>=3)}")
    print("\nTop 15 terms by mean delta AP:")
    print(f"{'go_id':<14}{'sup':>8}{'ap_base':>9}{'ap_fv':>9}{'delta':>9}{'pos/n':>8}")
    for r in rows[:15]:
        print(f"{r['go_id']:<14}{r['support_mean']:>8.0f}{r['ap_base']:>9.3f}{r['ap_fv']:>9.3f}{r['delta']:>+9.3f}{str(r['pos_seeds'])+'/'+str(r['n_seeds']):>8}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
