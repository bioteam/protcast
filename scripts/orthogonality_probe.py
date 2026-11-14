"""Orthogonality probe: can the ESM-C embedding predict structural features?

Low CV R2  -> structural feature is ORTHOGONAL to the embedding (worth pursuing).
High CV R2 -> redundant (ESM already encodes it; handcrafted struct FV would be dead too).

Uses pooled mean_max_std ESM embeddings (the production pooling) we already have,
plus AlphaFold-DB structures (downloaded) for a random sample of proteins.
Structural FV is computed from CA coordinates + pLDDT (B-factor col) only -- no DSSP.
"""
import os, sys, glob, pickle, random, urllib.request, urllib.error, json, csv
import numpy as np
from concurrent.futures import ThreadPoolExecutor

EMBDIR = sys.argv[1]
OUTDIR = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 1200
os.makedirs(OUTDIR, exist_ok=True)

print("Loading embeddings ...", flush=True)
emb = {}
for f in sorted(glob.glob(os.path.join(EMBDIR, "GO_*.pkl"))):
    d = pickle.load(open(f, "rb"))
    for pid, v in d.items():
        if pid not in emb:
            emb[pid] = np.asarray(v, dtype=np.float32)
pids_all = sorted(emb)
print(f"total proteins with embeddings: {len(pids_all)}  (emb dim {emb[pids_all[0]].shape[0]})", flush=True)

random.seed(0)
sample = random.sample(pids_all, min(N, len(pids_all)))


def dl(pid):
    out = os.path.join(OUTDIR, f"AF-{pid}-F1-model_v6.pdb")
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        return (pid, True)
    for ver in ("v6", "v5", "v4"):
        url = f"https://alphafold.ebi.ac.uk/files/AF-{pid}-F1-model_{ver}.pdb"
        try:
            urllib.request.urlretrieve(url, out)
            if os.path.getsize(out) > 1000:
                return (pid, True)
        except Exception:
            continue
    if os.path.exists(out):
        os.remove(out)
    return (pid, False)


print(f"Downloading {len(sample)} structures ...", flush=True)
ok = {}
with ThreadPoolExecutor(max_workers=16) as ex:
    for pid, good in ex.map(dl, sample):
        ok[pid] = good
have = [p for p in sample if ok[p]]
print(f"AlphaFold-DB coverage: {len(have)}/{len(sample)} = {len(have)/len(sample):.1%}", flush=True)


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
    return {
        "length": float(L),
        "Rg": Rg,
        "Rg_over_sqrtL": float(Rg / np.sqrt(L)),
        "asphericity": asph,
        "contact_density": cd,
        "rel_contact_order": rco,
        "end_to_end": float(np.linalg.norm(R[0] - R[-1])),
        "mean_plddt": float(pl.mean()),
        "frac_plddt_lt70": float((pl < 70).mean()),
        "frac_plddt_lt50": float((pl < 50).mean()),
    }


print("Featurizing ...", flush=True)
feats = {}
for pid in have:
    fv = featurize(os.path.join(OUTDIR, f"AF-{pid}-F1-model_v6.pdb"))
    if fv:
        feats[pid] = fv
fpids = list(feats)
keys = list(feats[fpids[0]].keys())
S = np.array([[feats[p][k] for k in keys] for p in fpids])
X = np.array([emb[p] for p in fpids])
print(f"regression matrix: X={X.shape}  S={S.shape}", flush=True)

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

print(f"\n{'structural feature':<20}{'CV R2 (ESM->feat)':>18}   verdict")
print("-" * 60)
rows = []
for j, k in enumerate(keys):
    y = S[:, j]
    model = make_pipeline(StandardScaler(), PCA(n_components=100, random_state=0),
                          RidgeCV(alphas=[0.1, 1, 10, 100, 1000]))
    r2 = float(cross_val_score(model, X, y, cv=5, scoring="r2").mean())
    verdict = "ORTHOGONAL" if r2 < 0.3 else ("partial" if r2 < 0.7 else "redundant")
    print(f"{k:<20}{r2:>18.3f}   {verdict}")
    rows.append({"feature": k, "cv_r2": round(r2, 4), "verdict": verdict})

with open(os.path.join(OUTDIR, "orthogonality_R2.csv"), "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=["feature", "cv_r2", "verdict"])
    w.writeheader(); w.writerows(rows)
print(f"\nn proteins in regression: {len(fpids)} | embedding dim {X.shape[1]} (PCA->100)")
print(f"Wrote {os.path.join(OUTDIR, 'orthogonality_R2.csv')}")
