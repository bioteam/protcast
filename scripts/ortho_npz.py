"""Orthogonality of a saved FV (npz: pids,S,names) vs an embedding (npz: pids,X).
CV R2 of embedding -> each FV dim (PCA-100 + RidgeCV)."""
import sys, csv
import numpy as np
EMB_NPZ, FV_NPZ, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
ze = np.load(EMB_NPZ, allow_pickle=True); Xe = ze["X"]; pe = ze["pids"]
emb = {p: Xe[i] for i, p in enumerate(pe)}
zf = np.load(FV_NPZ, allow_pickle=True); names = list(zf["names"]); Sf = zf["S"]; pf = zf["pids"]
fv = {p: Sf[i] for i, p in enumerate(pf)}
common = sorted(set(emb) & set(fv))
X = np.array([emb[p] for p in common]); S = np.array([fv[p] for p in common])
print(f"orthogonality probe: n={len(common)}  X={X.shape}  S={S.shape}")
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
print(f"\n{'USR feature':<14}{'CV R2':>8}  verdict")
print("-" * 36)
rows = []
for j, k in enumerate(names):
    model = make_pipeline(StandardScaler(), PCA(n_components=100, random_state=0),
                          RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
    r2 = float(cross_val_score(model, X, S[:, j], cv=5, scoring="r2").mean())
    v = "ORTHOGONAL" if r2 < 0.3 else ("partial" if r2 < 0.7 else "redundant")
    print(f"{k:<14}{r2:>8.3f}  {v}")
    rows.append({"feature": k, "cv_r2": round(r2, 4), "verdict": v})
with open(OUT, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=["feature", "cv_r2", "verdict"]); w.writeheader(); w.writerows(rows)
print(f"\nwrote {OUT}")
