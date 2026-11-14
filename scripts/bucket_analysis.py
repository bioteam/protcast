"""Bucket all 291 level-5 terms by has_site coverage (all-site / mixed / no-site)
and report support-weighted ΔAP and ΔFmax per bucket from active_site_perterm.csv.
Localizes WHERE the ESM+FV gain lives -- the has_site 'enzyme bit' can only vary in
mixed terms, so if the gain concentrates there it implicates the indicator, not
active-site chemistry."""
import os, sys, glob, pickle, csv
import numpy as np
from collections import defaultdict

WORK = os.environ["WORK"]
EMBDIR = os.path.join(WORK, "ProtCast", "ProtCastDataset/01-23-2026/mf_go_terms-level-5-mean_max_std")
# has_site coverage is defined by the full 22-dim FV regardless of which FV was trained
FV = os.path.join(WORK, "ProtCast", "active_site_fv.npz")
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(WORK, "ProtCast", "active_site_perterm.csv")
print(f"CSV: {CSV}")

z = np.load(FV, allow_pickle=True)
has = {p: float(s) for p, s in zip(z["pids"], z["S"][:, 21])}

bucket = {}  # go -> 'all'/'mixed'/'none'
for f in sorted(glob.glob(os.path.join(EMBDIR, "*.pkl"))):
    d = pickle.load(open(f, "rb"))
    members = [p for p in d if p in has]
    if not members:
        continue
    go = "GO_" + os.path.basename(f).split("_")[1].split(".")[0]
    fr = sum(has[p] for p in members) / len(members)
    bucket[go] = "all" if fr == 1.0 else ("none" if fr == 0.0 else "mixed")

# per-seed support-weighted deltas within each bucket
rows_by = {b: defaultdict(list) for b in ("all", "mixed", "none")}  # bucket -> seed -> [(sup,dap,dfm)]
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        b = bucket.get(r["go_id"])
        if b is None:
            continue
        sup = int(r["support"])
        dap = float(r["ap_fv"]) - float(r["ap_base"])
        dfm = float(r["fmax_fv"]) - float(r["fmax_base"])
        rows_by[b][int(r["seed"])].append((sup, dap, dfm))

n_terms = {b: sum(1 for g in bucket if bucket[g] == b) for b in ("all", "mixed", "none")}
print(f"{'bucket':<7} {'#terms':>6} {'mean dAP':>10} {'sd':>8} {'mean dFmax':>11} {'sd':>8}")
for b in ("all", "mixed", "none"):
    seed_ap, seed_fm = [], []
    for seed, lst in rows_by[b].items():
        wsup = sum(s for s, _, _ in lst)
        if not wsup:
            continue
        seed_ap.append(sum(s * d for s, d, _ in lst) / wsup)
        seed_fm.append(sum(s * d for s, _, d in lst) / wsup)
    if seed_ap:
        print(f"{b:<7} {n_terms[b]:>6} {np.mean(seed_ap):>+10.4f} {np.std(seed_ap):>8.4f} "
              f"{np.mean(seed_fm):>+11.4f} {np.std(seed_fm):>8.4f}")
    else:
        print(f"{b:<7} {n_terms[b]:>6}   (no rows)")
