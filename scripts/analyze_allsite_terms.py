"""Restrict the per-term FV results to the 64 GO terms where EVERY member has an
annotated active site (has_site==1). Within those terms has_site is constant, so any
ESM+FV gain is purely the 20-dim active-site AA composition.

Reports, over the all-site terms only:
  - support-weighted mean dAP and dFmax, averaged across seeds (paired)
  - per-seed support-weighted deltas
  - # terms improved vs hurt (by mean-over-seeds per-term delta)
  - paired t-test across the per-term mean deltas (vs 0)
"""
import os
import glob
import pickle
import csv
import numpy as np
from collections import defaultdict

WORK = os.environ["WORK"]
EMBDIR = os.path.join(WORK, "ProtCast",
                      "ProtCastDataset/01-23-2026/mf_go_terms-level-5-mean_max_std")
FV = os.path.join(WORK, "ProtCast", "active_site_fv.npz")
CSV = os.path.join(WORK, "ProtCast", "active_site_perterm.csv")

# --- identify the all-site terms (frac has_site == 1.0) ---
z = np.load(FV, allow_pickle=True)
has = {p: float(s) for p, s in zip(z["pids"], z["S"][:, 21])}
allsite = set()
term_n = {}
for f in sorted(glob.glob(os.path.join(EMBDIR, "*.pkl"))):
    d = pickle.load(open(f, "rb"))
    members = [p for p in d if p in has]
    if not members:
        continue
    go = "GO_" + os.path.basename(f).split("_")[1].split(".")[0]
    term_n[go] = len(members)
    if all(has[p] == 1.0 for p in members):
        allsite.add(go)
print(f"all-site terms (every member has_site==1): {len(allsite)}")

# --- load per-term CSV ---
by_seed = defaultdict(list)        # seed -> list of (go,support,d_ap,d_fmax)
per_term_ap = defaultdict(list)    # go -> [d_ap over seeds]
per_term_fmax = defaultdict(list)
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        go = r["go_id"]
        if go not in allsite:
            continue
        sup = int(r["support"])
        d_ap = float(r["ap_fv"]) - float(r["ap_base"])
        d_fm = float(r["fmax_fv"]) - float(r["fmax_base"])
        by_seed[int(r["seed"])].append((go, sup, d_ap, d_fm))
        per_term_ap[go].append(d_ap)
        per_term_fmax[go].append(d_fm)

if not by_seed:
    print("No rows for all-site terms in CSV -- has the per-term job finished?")
    raise SystemExit

print("\nPer-seed support-weighted deltas (all-site terms only):")
seed_ap, seed_fm = [], []
for seed in sorted(by_seed):
    rows = by_seed[seed]
    wsup = sum(s for _, s, _, _ in rows)
    wap = sum(s * d for _, s, d, _ in rows) / wsup
    wfm = sum(s * d for _, s, _, d in rows) / wsup
    seed_ap.append(wap); seed_fm.append(wfm)
    print(f"  seed {seed:>5}: terms={len(rows)}  dAP={wap:+.4f}  dFmax={wfm:+.4f}")

print(f"\nAcross seeds (support-weighted, all-site terms):")
print(f"  mean dAP   = {np.mean(seed_ap):+.4f}  (SD {np.std(seed_ap):.4f})")
print(f"  mean dFmax = {np.mean(seed_fm):+.4f}  (SD {np.std(seed_fm):.4f})")

# per-term mean-over-seeds delta
mean_ap = {g: np.mean(v) for g, v in per_term_ap.items()}
mean_fm = {g: np.mean(v) for g, v in per_term_fmax.items()}
imp_ap = sum(1 for v in mean_ap.values() if v > 0)
imp_fm = sum(1 for v in mean_fm.values() if v > 0)
print(f"\nPer-term (mean over seeds), n={len(mean_ap)} all-site terms:")
print(f"  AP improved:   {imp_ap}/{len(mean_ap)}")
print(f"  Fmax improved: {imp_fm}/{len(mean_fm)}")

# paired t-test of per-term mean deltas vs 0
try:
    from scipy import stats
    for name, dd in [("dAP", list(mean_ap.values())), ("dFmax", list(mean_fm.values()))]:
        t, p = stats.ttest_1samp(dd, 0.0)
        print(f"  {name}: mean={np.mean(dd):+.4f}  t={t:+.2f}  p={p:.3g}")
except Exception as e:
    print("  (scipy unavailable for t-test:", e, ")")

# biggest movers
print("\nTop 10 all-site terms by mean dFmax:")
for g in sorted(mean_fm, key=mean_fm.get, reverse=True)[:10]:
    print(f"  {g}  n={term_n.get(g,'?'):>5}  dFmax={mean_fm[g]:+.4f}  dAP={mean_ap[g]:+.4f}")
print("Bottom 5 all-site terms by mean dFmax:")
for g in sorted(mean_fm, key=mean_fm.get)[:5]:
    print(f"  {g}  n={term_n.get(g,'?'):>5}  dFmax={mean_fm[g]:+.4f}  dAP={mean_ap[g]:+.4f}")
