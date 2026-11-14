"""Orthogonality probe, split into stages to fit TACC rules
(containers only on compute nodes; downloads only on login nodes).

  stage=prep    : load embeddings, sample N pids -> sample_pids.txt + sample_emb.npz
  stage=analyze : featurize downloaded structures + ridge-CV regress emb->feature
"""
import os, sys, glob, pickle, random, csv
import numpy as np

stage = sys.argv[1]
OUTDIR = sys.argv[2]


def prep(embdir, n):
    print("Loading embeddings ...", flush=True)
    emb = {}
    for f in sorted(glob.glob(os.path.join(embdir, "GO_*.pkl"))):
        d = pickle.load(open(f, "rb"))
        for pid, v in d.items():
            if pid not in emb:
                emb[pid] = np.asarray(v, dtype=np.float32)
    pids_all = sorted(emb)
    print(f"total proteins: {len(pids_all)}  emb dim {emb[pids_all[0]].shape[0]}")
    random.seed(0)
    sample = random.sample(pids_all, min(n, len(pids_all)))
    with open(os.path.join(OUTDIR, "sample_pids.txt"), "w") as fo:
        fo.write("\n".join(sample) + "\n")
    X = np.vstack([emb[p] for p in sample])
    np.savez(os.path.join(OUTDIR, "sample_emb.npz"),
             pids=np.array(sample), X=X.astype(np.float32))
    print(f"wrote {len(sample)} pids + emb {X.shape}")


def featurize(pdb):
    R, b = [], []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            R.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
            b.append(float(line[60:66]))
    if len(R) < 10:
        return None
    R = np.array(R); pl = np.array(b); L = len(R)
    cm = R.mean(0)
    Rg = float(np.sqrt(((R - cm) ** 2).sum(1).mean()))
    cov = np.cov((R - cm).T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    asph = float((ev[0] - 0.5 * (ev[1] + ev[2])) / ev.sum())
    dist = np.sqrt(((R[:, None, :] - R[None, :, :]) ** 2).sum(-1))
    sep = np.abs(np.subtract.outer(np.arange(L), np.arange(L)))
    mask = (dist < 8.0) & (sep >= 3)
    nc = mask.sum() / 2.0
    cd = float(nc / L)
    rco = float((sep[mask].sum() / 2.0) / (L * nc)) if nc > 0 else 0.0
    return {"length": float(L), "Rg": Rg, "Rg_over_sqrtL": float(Rg / np.sqrt(L)),
            "asphericity": asph, "contact_density": cd, "rel_contact_order": rco,
            "end_to_end": float(np.linalg.norm(R[0] - R[-1])),
            "mean_plddt": float(pl.mean()), "frac_plddt_lt70": float((pl < 70).mean()),
            "frac_plddt_lt50": float((pl < 50).mean())}


def analyze():
    z = np.load(os.path.join(OUTDIR, "sample_emb.npz"), allow_pickle=True)
    pids = list(z["pids"]); X_all = z["X"]
    emb = {p: X_all[i] for i, p in enumerate(pids)}
    feats = {}
    for p in pids:
        pdb = os.path.join(OUTDIR, f"AF-{p}-F1-model_v6.pdb")
        if os.path.exists(pdb) and os.path.getsize(pdb) > 1000:
            fv = featurize(pdb)
            if fv:
                feats[p] = fv
    print(f"coverage with valid structures: {len(feats)}/{len(pids)} = {len(feats)/len(pids):.1%}")
    fpids = list(feats); keys = list(feats[fpids[0]].keys())
    S = np.array([[feats[p][k] for k in keys] for p in fpids])
    X = np.array([emb[p] for p in fpids])
    print(f"regression matrix: X={X.shape}  S={S.shape}")

    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline

    print(f"\n{'structural feature':<20}{'CV R2 (ESM->feat)':>18}   verdict")
    print("-" * 60)
    rows = []
    for j, k in enumerate(keys):
        model = make_pipeline(StandardScaler(), PCA(n_components=100, random_state=0),
                              RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
        r2 = float(cross_val_score(model, X, S[:, j], cv=5, scoring="r2").mean())
        verdict = "ORTHOGONAL" if r2 < 0.3 else ("partial" if r2 < 0.7 else "redundant")
        print(f"{k:<20}{r2:>18.3f}   {verdict}")
        rows.append({"feature": k, "cv_r2": round(r2, 4), "verdict": verdict})
    with open(os.path.join(OUTDIR, "orthogonality_R2.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=["feature", "cv_r2", "verdict"])
        w.writeheader(); w.writerows(rows)
    print(f"\nn={len(fpids)}  emb dim {X.shape[1]} (PCA->100)\nWrote orthogonality_R2.csv")


if stage == "prep":
    prep(sys.argv[3], int(sys.argv[4]))
elif stage == "analyze":
    analyze()
