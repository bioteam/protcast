import csv, os
import numpy as np
from collections import defaultdict

WORK = os.environ["WORK"]
os.chdir(os.path.join(WORK, "ProtCast"))


def load(fn):
    d = defaultdict(list)
    for r in csv.DictReader(open(fn)):
        d[r["go_id"]].append(float(r["fmax_fv"]) - float(r["fmax_base"]))
    return d


full = load("active_site_perterm.csv")   # 22-dim: composition + site_overlap + has_site
aa = load("as_aa_perterm.csv")           # 20-dim: composition only

tops = ["GO_0003862", "GO_0008839", "GO_0070402", "GO_0004810", "GO_0004585",
        "GO_0004765", "GO_0004832", "GO_0004764", "GO_0035597", "GO_0004347"]

hdr = "{:<12}{:>22}{:>24}".format("term", "full dFmax (pos/12)", "comp-only dFmax (pos/12)")
print(hdr)
for t in tops:
    f = full.get(t, [])
    a = aa.get(t, [])
    fm = np.mean(f) if f else float("nan")
    fp = sum(1 for x in f if x > 0)
    am = np.mean(a) if a else float("nan")
    ap = sum(1 for x in a if x > 0)
    print("{:<12}{:>+14.4f} ({:>2}/{:<2}){:>+16.4f} ({:>2}/{:<2})".format(
        t, fm, fp, len(f), am, ap, len(a)))
