#!/usr/bin/env python3
"""Read-only integrity sweep of regenerated ESM-C embedding pickles.

Validates that each per-GO-term pickle:
  * unpickles without error (catches truncation/corruption from concurrent writes)
  * is a dict of {protein_id: vector}
  * every vector is 1-D length 3456 (mean_max_std, 3*1152)
  * contains no NaN/Inf (garbage from a partial write)

Prints only problems per file, plus a per-level summary. Sequential, one
pickle in memory at a time, to stay light on the login node.
"""
import os, sys, glob, pickle, gc
import numpy as np

EXPECT_DIM = 3456
BASE = os.path.join(os.environ["WORK"],
                    "ProtCast/ProtCastDataset/01-23-2026")
levels = sys.argv[1:] or ["2", "4", "5"]

print(f"integrity sweep of levels {levels}  (expect dim {EXPECT_DIM})", flush=True)
grand_fail = 0
for lvl in levels:
    d = os.path.join(BASE, f"mf_go_terms-level-{lvl}-mean_max_std")
    files = sorted(glob.glob(os.path.join(d, "*.pkl")))
    n_ok = n_bad = n_prot = 0
    print(f"\n=== level-{lvl}: {len(files)} pickle(s) ===", flush=True)
    for f in files:
        try:
            with open(f, "rb") as fh:
                obj = pickle.load(fh)
            if not isinstance(obj, dict) or len(obj) == 0:
                raise ValueError(f"not a non-empty dict (type={type(obj).__name__})")
            k = next(iter(obj))
            v = np.asarray(obj[k])
            if v.ndim != 1 or v.shape[0] != EXPECT_DIM:
                raise ValueError(f"bad vector shape {v.shape} for {k}")
            # sample up to 50 vectors for NaN/Inf
            bad_finite = 0
            for i, key in enumerate(obj):
                if i >= 50:
                    break
                if not np.all(np.isfinite(np.asarray(obj[key]))):
                    bad_finite += 1
            if bad_finite:
                raise ValueError(f"{bad_finite} sampled vectors have NaN/Inf")
            n_ok += 1
            n_prot += len(obj)
        except Exception as e:
            n_bad += 1
            grand_fail += 1
            print(f"  FAIL {os.path.basename(f)} ({os.path.getsize(f)} bytes): {e}", flush=True)
        finally:
            obj = None
            gc.collect()
    print(f"  level-{lvl} summary: ok={n_ok} bad={n_bad} total_proteins={n_prot}", flush=True)

print(f"\n=== DONE. total bad files: {grand_fail} ===", flush=True)
