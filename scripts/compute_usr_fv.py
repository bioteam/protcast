"""12-dim USR (Ultrafast Shape Recognition) over all heavy atoms of each structure.
4 reference points (centroid, atom nearest centroid, atom farthest from centroid,
atom farthest from that) x 3 moments (mean, std, skew) of the distance distribution.
Rotation/translation invariant, fixed length. Saves usr_fv.npz in STRUCTDIR."""
import os, sys, glob
import numpy as np

STRUCTDIR = sys.argv[1]
names = [f"{r}_{m}" for r in ("ctd", "cst", "fct", "ftf") for m in ("mean", "std", "skew")]


def parse_atoms(pdb):
    R = []
    for line in open(pdb):
        if line.startswith("ATOM"):
            R.append((float(line[30:38]), float(line[38:46]), float(line[46:54])))
    return np.asarray(R) if len(R) >= 10 else None


def usr(C):
    ctd = C.mean(0)
    d = np.linalg.norm(C - ctd, axis=1)
    cst = C[d.argmin()]; fct = C[d.argmax()]
    ftf = C[np.linalg.norm(C - fct, axis=1).argmax()]
    out = []
    for ref in (ctd, cst, fct, ftf):
        dd = np.linalg.norm(C - ref, axis=1)
        m = dd.mean(); s = dd.std()
        sk = float((((dd - m) ** 3).mean()) / (s ** 3)) if s > 1e-8 else 0.0
        out += [float(m), float(s), sk]
    return np.array(out, dtype=np.float32)


pdbs = sorted(glob.glob(os.path.join(STRUCTDIR, "AF-*-F1-model_v6.pdb")))
print(f"USR over {len(pdbs)} structures (all heavy atoms) ...", flush=True)
pids, rows, bad = [], [], 0
for k, pdb in enumerate(pdbs):
    if os.path.getsize(pdb) <= 1000:
        continue
    C = parse_atoms(pdb)
    if C is None:
        bad += 1; continue
    fv = usr(C)
    if not np.all(np.isfinite(fv)):
        bad += 1; continue
    pids.append(os.path.basename(pdb).split("-")[1]); rows.append(fv)
    if (k + 1) % 5000 == 0:
        print(f"  {k+1}/{len(pdbs)}", flush=True)
S = np.vstack(rows)
np.savez(os.path.join(STRUCTDIR, "usr_fv.npz"), pids=np.array(pids), S=S, names=np.array(names))
print(f"saved {S.shape} -> usr_fv.npz ({bad} skipped)")
print("means:", dict(zip(names, S.mean(0).round(2))))
print("stds :", dict(zip(names, S.std(0).round(2))))
