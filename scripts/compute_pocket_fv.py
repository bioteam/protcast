"""Pocket FV via fpocket. Per structure: run fpocket, parse top-3 pockets by Score,
extract [Volume, Druggability Score, Total SASA, Hydrophobicity, Polarity] each,
plus n_pockets -> 16-dim FV. Runs fpocket in a /tmp scratch dir and deletes the
_out folder immediately (no inode explosion). Parallel via multiprocessing."""
import os, sys, glob, shutil, tempfile, subprocess
import numpy as np
from multiprocessing import Pool

STRUCTDIR = sys.argv[1]
LIMIT = int(os.environ.get("LIMIT", "0"))
NPROC = int(os.environ.get("NPROC", "48"))
FPOCKET = os.path.expanduser("~/git/fpocket/bin/fpocket")
KEYS = ["Volume", "Druggability Score", "Total SASA", "Hydrophobicity score", "Polarity score"]
SHORT = ["vol", "drug", "sasa", "hydro", "polar"]
NTOP = 3
names = [f"p{i+1}_{s}" for i in range(NTOP) for s in SHORT] + ["n_pockets"]


def parse_info(path):
    pockets, cur = [], None
    for line in open(path):
        if line.startswith("Pocket"):
            cur = {}; pockets.append(cur)
        elif cur is not None and ":" in line:
            k, _, v = line.strip().partition(":")
            try:
                cur[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return pockets


def feat(pdb):
    pid = os.path.basename(pdb).split("-")[1]
    tmp = tempfile.mkdtemp(dir="/tmp")
    try:
        local = os.path.join(tmp, "s.pdb")
        shutil.copy(pdb, local)
        subprocess.run([FPOCKET, "-f", local], cwd=tmp,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        info = os.path.join(tmp, "s_out", "s_info.txt")
        if not os.path.exists(info):
            return (pid, None)
        pockets = parse_info(info)
        pockets.sort(key=lambda p: p.get("Score", 0.0), reverse=True)
        fv = []
        for i in range(NTOP):
            if i < len(pockets):
                fv += [pockets[i].get(k, 0.0) for k in KEYS]
            else:
                fv += [0.0] * len(KEYS)
        fv.append(float(len(pockets)))
        return (pid, np.array(fv, dtype=np.float32))
    except Exception:
        return (pid, None)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


pdbs = sorted(glob.glob(os.path.join(STRUCTDIR, "AF-*-F1-model_v6.pdb")))
if LIMIT:
    pdbs = pdbs[:LIMIT]
print(f"fpocket over {len(pdbs)} structures, {NPROC} procs ...", flush=True)
pids, rows, bad = [], [], 0
with Pool(NPROC) as pool:
    for j, (pid, fv) in enumerate(pool.imap_unordered(feat, pdbs, chunksize=16)):
        if fv is None:
            bad += 1; continue
        pids.append(pid); rows.append(fv)
        if (j + 1) % 5000 == 0:
            print(f"  {j+1}/{len(pdbs)}", flush=True)
S = np.vstack(rows)
out = os.path.join(STRUCTDIR, "pocket_fv.npz" if not LIMIT else "pocket_fv_TEST.npz")
np.savez(out, pids=np.array(pids), S=S, names=np.array(names))
print(f"saved {S.shape} -> {os.path.basename(out)} ({bad} failed)")
print("means:", dict(zip(names, S.mean(0).round(2))))
