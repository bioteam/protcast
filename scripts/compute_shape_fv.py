"""Compute the 4-dim shape FV for every downloaded AlphaFold structure.

Features (all chosen because ESM-C does NOT already encode them; orthogonality
probe R2 < ~0.6): asphericity, end_to_end/Rg, rel_contact_order, Rg/sqrt(L).
Saves shape_fv.npz  (pids, S[n,4]) in STRUCTDIR.
"""
import os, sys, glob
import numpy as np

STRUCTDIR = sys.argv[1]
FEAT_NAMES = ["asphericity", "end_to_end_over_Rg", "rel_contact_order", "Rg_over_sqrtL"]


def featurize(pdb):
    R, idx = [], []
    for line in open(pdb):
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            R.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    if len(R) < 10:
        return None
    R = np.asarray(R); L = len(R); cm = R.mean(0)
    Rg = float(np.sqrt(((R - cm) ** 2).sum(1).mean()))
    if Rg == 0:
        return None
    cov = np.cov((R - cm).T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    asph = float((ev[0] - 0.5 * (ev[1] + ev[2])) / ev.sum())
    # contacts (chunked to bound memory for long chains)
    sep_sum = 0.0; nc = 0
    for i in range(0, L, 512):
        block = R[i:i + 512]
        d = np.sqrt(((block[:, None, :] - R[None, :, :]) ** 2).sum(-1))
        ii = (np.arange(block.shape[0])[:, None] + i)
        jj = np.arange(L)[None, :]
        sep = np.abs(ii - jj)
        m = (d < 8.0) & (sep >= 3)
        nc += int(m.sum())
        sep_sum += float(sep[m].sum())
    nc /= 2.0; sep_sum /= 2.0
    rco = float(sep_sum / (L * nc)) if nc > 0 else 0.0
    e2e = float(np.linalg.norm(R[0] - R[-1]))
    return np.array([asph, e2e / Rg, rco, Rg / np.sqrt(L)], dtype=np.float32)


pids, rows = [], []
pdbs = sorted(glob.glob(os.path.join(STRUCTDIR, "AF-*-F1-model_v6.pdb")))
print(f"featurizing {len(pdbs)} structures ...", flush=True)
bad = 0
for k, pdb in enumerate(pdbs):
    if os.path.getsize(pdb) <= 1000:
        continue
    fv = featurize(pdb)
    if fv is None or not np.all(np.isfinite(fv)):
        bad += 1
        continue
    pid = os.path.basename(pdb).split("-")[1]  # AF-<pid>-F1-model_v6.pdb
    pids.append(pid); rows.append(fv)
    if (k + 1) % 5000 == 0:
        print(f"  {k+1}/{len(pdbs)}", flush=True)

S = np.vstack(rows)
np.savez(os.path.join(STRUCTDIR, "shape_fv.npz"), pids=np.array(pids), S=S,
         names=np.array(FEAT_NAMES))
print(f"saved {S.shape} shape FVs ({bad} skipped) -> shape_fv.npz")
print("feature means:", dict(zip(FEAT_NAMES, S.mean(0).round(3))))
print("feature stds :", dict(zip(FEAT_NAMES, S.std(0).round(3))))
