#!/usr/bin/env python3
"""Re-run fpocket on every AF structure and persist the amino-acid composition
and residue list of every detected pocket.

The earlier pocket-FV run (compute_pocket_fv.py) computed only the 16 geometric
descriptors and discarded fpocket's output, so the pocket *chemistry* (which
residues / amino acids line each pocket) was lost. This pass saves it.

Output: $WORK/ProtCast/pocket_residues.jsonl, one JSON object per protein per
pocket:
  {"id": "AF-..-F1-model_v6", "pocket": 1, "score": 0.42, "n_res": 17,
   "residues": [["A","123","HIS"], ...], "aa": {"HIS": 2, "ASP": 1, ...}}
Proteins with no detected pocket get {"id": ..., "n_pockets": 0}.

Resumable: any protein id already present in the output file is skipped, so a
timed-out job can simply be resubmitted.
"""
import os
import sys
import glob
import json
import shutil
import tempfile
import subprocess
from collections import Counter
from multiprocessing import Pool

WORK = os.environ["WORK"]
STRUCT_DIR = os.path.join(WORK, "ProtCast", "structures")
OUT = os.path.join(WORK, "ProtCast", "pocket_residues.jsonl")
FPOCKET = os.path.expanduser("~/git/fpocket/bin/fpocket")
NPROC = int(os.environ.get("NPROC", "48"))

AA3 = {"ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}


def parse_info_scores(info_path):
    """Map 1-based pocket number -> fpocket Score from <name>_info.txt."""
    scores = {}
    cur = None
    with open(info_path) as fh:
        for raw in fh:
            line = raw.strip()
            if line.lower().startswith("pocket"):
                parts = line.split()
                cur = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            elif cur is not None and ":" in line:
                key, _, val = line.partition(":")
                if key.strip().lower() == "score":
                    try:
                        scores[cur] = float(val.strip())
                    except ValueError:
                        pass
    return scores


def parse_pocket_atoms(atm_path):
    """Unique residues (chain, resSeq) -> resName lining a pocket."""
    residues = {}
    with open(atm_path) as fh:
        for line in fh:
            if line.startswith(("ATOM", "HETATM")):
                resname = line[17:20].strip()
                chain = line[21].strip() or "A"
                resseq = line[22:26].strip()
                residues[(chain, resseq)] = resname
    return residues


def process(pdb):
    name = os.path.basename(pdb)[:-4]  # strip .pdb
    tmp = tempfile.mkdtemp(prefix="fp_")
    try:
        local = os.path.join(tmp, name + ".pdb")
        shutil.copyfile(pdb, local)
        subprocess.run([FPOCKET, "-f", local], cwd=tmp,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=600)
        out_dir = os.path.join(tmp, name + "_out")
        info = os.path.join(out_dir, name + "_info.txt")
        scores = parse_info_scores(info) if os.path.exists(info) else {}
        pk_dir = os.path.join(out_dir, "pockets")
        rows = []
        if os.path.isdir(pk_dir):
            atm_files = sorted(
                glob.glob(os.path.join(pk_dir, "pocket*_atm.pdb")),
                key=lambda p: int(os.path.basename(p)[6:].split("_")[0]))
            for atm in atm_files:
                idx = int(os.path.basename(atm)[6:].split("_")[0])  # fpocket files are 1-based
                resids = parse_pocket_atoms(atm)
                aa = Counter(rn for rn in resids.values() if rn in AA3)
                rows.append({
                    "id": name,
                    "pocket": idx,                # 1-based, matches _info.txt
                    "score": scores.get(idx),
                    "n_res": len(resids),
                    "residues": [[c, s, resids[(c, s)]] for (c, s) in resids],
                    "aa": dict(aa),
                })
        return name, rows
    except Exception as e:  # noqa: BLE001 - record and continue
        return name, [{"id": name, "error": repr(e)}]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    pdbs = sorted(glob.glob(os.path.join(STRUCT_DIR, "*.pdb")))
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass
    todo = [p for p in pdbs if os.path.basename(p)[:-4] not in done]
    print(f"{len(pdbs)} structures, {len(done)} done, {len(todo)} remaining",
          flush=True)
    with open(OUT, "a", buffering=1) as out, Pool(NPROC) as pool:
        n = 0
        for name, rows in pool.imap_unordered(process, todo, chunksize=8):
            if rows:
                for row in rows:
                    out.write(json.dumps(row) + "\n")
            else:
                out.write(json.dumps({"id": name, "n_pockets": 0}) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"{n}/{len(todo)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
