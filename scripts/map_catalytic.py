"""Stratify per-term pocket-AA ΔAP by GO-MF functional class.

Parses go.obo (is_a DAG), labels each tested GO term as catalytic (descendant of
catalytic activity GO:0003824) and assigns an EC-like subclass, then tests
whether the pocket-AA FV helps a chemistry-dependent (enzyme) subset more than
the rest.

Honest metric (same as the GAAC probe): per SEED, the support-weighted mean ΔAP
across the terms in a class; then a t-test across seeds. Counting low-support
terms equally (unweighted sign test) is performance-irrelevant, so it's reported
only for reference.

Inputs:  --perterm <long CSV: go_id,seed,support,ap_base,ap_fv>  --obo <go.obo>
Output:  --out <json>   plus a printed table.
"""
import os
import sys
import csv
import json
import argparse
from collections import defaultdict
import numpy as np

CATALYTIC = "GO:0003824"
# EC-like top classes (checked in order); first match wins.
SUBCLASSES = [
    ("oxidoreductase", "GO:0016491"),
    ("transferase",    "GO:0016740"),
    ("hydrolase",      "GO:0016787"),
    ("lyase",          "GO:0016829"),
    ("isomerase",      "GO:0016853"),
    ("ligase",         "GO:0016874"),
    ("translocase",    "GO:0140657"),
]
# Non-catalytic groupings (checked in order).
NONCAT = [
    ("transporter", "GO:0005215"),
    ("binding",     "GO:0005488"),
]


def parse_obo(path):
    """Return child -> set(is_a parents)."""
    parents = defaultdict(set)
    cur = None
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if line == "[Term]":
                cur = None
            elif line.startswith("id: GO:"):
                cur = line[4:].strip()
            elif line.startswith("is_a:") and cur:
                # "is_a: GO:0016740 ! transferase activity"
                parents[cur].add(line.split()[1])
    return parents


def ancestors(go, parents, cache):
    if go in cache:
        return cache[go]
    seen = set()
    stack = list(parents.get(go, ()))
    while stack:
        p = stack.pop()
        if p not in seen:
            seen.add(p)
            stack.extend(parents.get(p, ()))
    cache[go] = seen
    return seen


def classify(go, parents, cache):
    anc = ancestors(go, parents, cache) | {go}
    is_cat = CATALYTIC in anc
    if is_cat:
        for name, gid in SUBCLASSES:
            if gid in anc:
                return "catalytic", name
        return "catalytic", "other_catalytic"
    for name, gid in NONCAT:
        if gid in anc:
            return "noncatalytic", name
    return "noncatalytic", "other_noncat"


def seed_weighted_deltas(rows_by_seed):
    """rows_by_seed: seed -> list of (support, ap_base, ap_fv). Return per-seed wtd dAP."""
    vals = []
    for seed in sorted(rows_by_seed):
        rs = rows_by_seed[seed]
        wsup = sum(s for s, _, _ in rs)
        if wsup == 0:
            continue
        vals.append(sum(s * (f - b) for s, b, f in rs) / wsup)
    return np.array(vals)


def summarize(vals):
    n = len(vals)
    if n == 0:
        return {"n_seeds": 0}
    mean = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / np.sqrt(n)) if sd > 0 else float("nan")
    return {"n_seeds": n, "mean_wtd_dAP": mean, "sd": sd,
            "t": t, "pos_seeds": int((vals > 0).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perterm", required=True)
    ap.add_argument("--obo", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    parents = parse_obo(args.obo)
    cache = {}

    # group rows: (group, subclass) and per term
    rows = list(csv.DictReader(open(args.perterm)))
    # normalize go ids GO_xxx -> GO:xxx
    for r in rows:
        r["go_id"] = r["go_id"].replace("GO_", "GO:")

    term_class = {}
    for r in rows:
        g = r["go_id"]
        if g not in term_class:
            term_class[g] = classify(g, parents, cache)

    # accumulate by group and subclass: class_key -> seed -> [(sup,b,f)]
    def acc(keyfn):
        d = defaultdict(lambda: defaultdict(list))
        nterms = defaultdict(set)
        for r in rows:
            k = keyfn(term_class[r["go_id"]])
            d[k][int(r["seed"])].append(
                (int(r["support"]), float(r["ap_base"]), float(r["ap_fv"])))
            nterms[k].add(r["go_id"])
        return d, nterms

    grp, grp_terms = acc(lambda c: c[0])           # catalytic / noncatalytic
    sub, sub_terms = acc(lambda c: f"{c[0]}:{c[1]}")  # subclass
    alld, all_terms = acc(lambda c: "ALL")

    result = {"groups": {}, "subclasses": {}, "all": {}}

    def fill(dest, d, nterms):
        for k in sorted(d):
            s = summarize(seed_weighted_deltas(d[k]))
            s["n_terms"] = len(nterms[k])
            dest[k] = s

    fill(result["all"], alld, all_terms)
    fill(result["groups"], grp, grp_terms)
    fill(result["subclasses"], sub, sub_terms)

    json.dump(result, open(args.out, "w"), indent=2)

    def line(name, s):
        if s.get("n_seeds", 0) == 0:
            print(f"  {name:28s}  (no data)")
            return
        t = s["t"]
        flag = "  *" if (t == t and abs(t) >= 2.0) else ""
        print(f"  {name:28s}  terms={s['n_terms']:4d}  seeds={s['n_seeds']:2d}  "
              f"wtd dAP={s['mean_wtd_dAP']:+.4f}  SD={s['sd']:.4f}  "
              f"t={t:+.2f}  pos={s['pos_seeds']}/{s['n_seeds']}{flag}")

    print("\n===== ALL TERMS =====")
    for k, s in result["all"].items():
        line(k, s)
    print("\n===== CATALYTIC vs NON-CATALYTIC =====")
    for k, s in result["groups"].items():
        line(k, s)
    print("\n===== EC-LIKE SUBCLASSES =====")
    for k, s in result["subclasses"].items():
        line(k, s)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
