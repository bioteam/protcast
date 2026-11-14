"""Per-term consistency summary for a single perterm CSV (go_id,seed,support,
ap_base,ap_fv,fmax_base,fmax_fv). Reports how broad/consistent the ESM+FV gain is."""
import sys, csv
import numpy as np
from collections import defaultdict

CSV = sys.argv[1]
dfm = defaultdict(list)   # go -> [dFmax per seed]
sup = {}
for r in csv.DictReader(open(CSV)):
    d = float(r["fmax_fv"]) - float(r["fmax_base"])
    dfm[r["go_id"]].append(d)
    sup[r["go_id"]] = int(r["support"])

terms = list(dfm)
mean = {g: np.mean(v) for g, v in dfm.items()}
posfrac = {g: np.mean([1 if x > 0 else 0 for x in dfm[g]]) for g in terms}  # frac seeds positive
n = len(terms)
print(f"terms: {n}")
print(f"mean-over-seeds dFmax > 0:        {sum(1 for g in terms if mean[g] > 0)}/{n}")
print(f"consistently positive (>=9/12):   {sum(1 for g in terms if posfrac[g] >= 0.75)}/{n}")
print(f"consistently negative (<=3/12):   {sum(1 for g in terms if posfrac[g] <= 0.25)}/{n}")
# support-weighted mean of per-term mean deltas
ws = sum(sup.values())
wmean = sum(sup[g] * mean[g] for g in terms) / ws
print(f"support-weighted mean dFmax:      {wmean:+.4f}")
print(f"unweighted mean of per-term means:{np.mean(list(mean.values())):+.4f}")
print("\nTop 10 terms by mean dFmax (with seeds-positive/12):")
for g in sorted(terms, key=mean.get, reverse=True)[:10]:
    npos = sum(1 for x in dfm[g] if x > 0)
    print(f"  {g}  n={sup[g]:>4}  dFmax={mean[g]:+.4f}  pos={npos}/{len(dfm[g])}")
