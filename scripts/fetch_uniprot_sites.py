#!/usr/bin/env python3
"""Fetch UniProt ACT_SITE + BINDING residue positions for all FV accessions.

Login-node only (compute nodes have no internet). Reads accessions from
pocket_aa_fv.npz (pids), queries UniProtKB REST in batches, parses residue
positions (UniProt numbering == AF-model resseq), writes:
  $WORK/ProtCast/uniprot_sites.json  ->  {accession: {"act":[..], "binding":[..]}}
Resumable: accessions already in the output file are skipped.
"""
import os
import re
import json
import time
import urllib.request
import urllib.parse
import numpy as np

WORK = os.environ["WORK"]
PIDS = os.path.join(WORK, "ProtCast", "pocket_aa_fv.npz")
OUT = os.path.join(WORK, "ProtCast", "uniprot_sites.json")
BATCH = 80
BASE = "https://rest.uniprot.org/uniprotkb/search"

ACT_RE = re.compile(r"ACT_SITE (\d+)")
BIND_RE = re.compile(r"BINDING (\d+)(?:\.\.(\d+))?")


def fetch(accs):
    q = " OR ".join("accession:%s" % a for a in accs)
    params = urllib.parse.urlencode({
        "query": "(%s)" % q,
        "fields": "accession,ft_act_site,ft_binding",
        "format": "tsv",
        "size": len(accs),
    })
    url = "%s?%s" % (BASE, params)
    for attempt in range(5):
        try:
            return urllib.request.urlopen(url, timeout=60).read().decode()
        except Exception as e:
            print("  retry %d: %r" % (attempt, e), flush=True)
            time.sleep(3 * (attempt + 1))
    return ""


def parse_line(line):
    parts = line.split("\t")
    if len(parts) < 3:
        return None
    acc, act_col, bind_col = parts[0], parts[1], parts[2]
    act = [int(m) for m in ACT_RE.findall(act_col)]
    binding = []
    for a, b in BIND_RE.findall(bind_col):
        if b:
            binding.extend(range(int(a), int(b) + 1))
        else:
            binding.append(int(a))
    return acc, sorted(set(act)), sorted(set(binding))


def main():
    z = np.load(PIDS, allow_pickle=True)
    accs = [str(p) for p in z["pids"]]
    sites = {}
    if os.path.exists(OUT):
        sites = json.load(open(OUT))
    todo = [a for a in accs if a not in sites]
    print("%d accessions, %d done, %d to fetch" % (len(accs), len(sites), len(todo)), flush=True)

    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        txt = fetch(chunk)
        seen = set()
        for line in txt.splitlines()[1:]:  # skip header
            r = parse_line(line)
            if r:
                acc, act, binding = r
                sites[acc] = {"act": act, "binding": binding}
                seen.add(acc)
        # accessions with no UniProt row (obsolete/demerged) -> empty record
        for a in chunk:
            sites.setdefault(a, {"act": [], "binding": []})
        if (i // BATCH) % 25 == 0:
            json.dump(sites, open(OUT, "w"))
            print("  %d/%d fetched (%d with ACT_SITE)" % (
                min(i + BATCH, len(todo)), len(todo),
                sum(1 for v in sites.values() if v["act"])), flush=True)
        time.sleep(0.2)

    json.dump(sites, open(OUT, "w"))
    n_act = sum(1 for v in sites.values() if v["act"])
    n_any = sum(1 for v in sites.values() if v["act"] or v["binding"])
    print("DONE: %d accessions, %d with ACT_SITE, %d with any site -> %s" % (
        len(sites), n_act, n_any, OUT), flush=True)


if __name__ == "__main__":
    main()
