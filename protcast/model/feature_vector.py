import os
import sys
import tempfile
from ifeatpro.features import get_feature


def get_ifeatpro_features(alg, seqs):
    """get_ifeatpro_features
    Returns a list of arrays or "feature vectors" using ifeatpro. ifeatpro
    creates a CSV file given a fasta file. The algorithms are:

    aac
    apaac
    cksaagp
    cksaap
    ctdc
    ctdd
    ctdt
    ctriad
    dde
    dpc
    gaac
    gdpc
    geary
    gtpc
    ksctriad
    moran
    nmbroto
    paac
    qsorder
    socnumber
    tpc

    Parameters
    ----------
    alg: str
        Algorithm name, e.g. 'ctriad'
    seqs: dict
        Key is protein id, value is protein sequence

    Returns
    -------
    List of lists of floats
    """
    algs = [
        "aac",
        "apaac",
        "cksaagp",
        "cksaap",
        "ctdc",
        "ctdd",
        "ctdt",
        "ctriad",
        "dde",
        "dpc",
        "gaac",
        "gdpc",
        "geary",
        "gtpc",
        "ksctriad",
        "moran",
        "nmbroto",
        "paac",
        "qsorder",
        "socnumber",
        "tpc",
    ]
    if alg not in algs:
        sys.exit(f"Algorithm {alg} not part of ifeatpro")

    ids = list()
    features = list()

    for pid, seq in seqs.items():
        tmpdir = tempfile.TemporaryDirectory()
        tmpfasta = tempfile.NamedTemporaryFile()
        # Write fasta file
        with open(tmpfasta.name, "w") as f:
            f.write(">" + pid + "\n" + str(seq.seq) + "\n")

        # The try is necessary in case the input sequence is too
        # short for a given algorithm
        try:
            get_feature(tmpfasta.name, alg, tmpdir.name)
            with open(os.path.join(tmpdir.name, alg + ".csv")) as f:
                for line in f.readlines():
                    arr = line.rstrip().split(",")
                    # Skip the first column which contains the id
                    vals = [float(x) for x in arr[1:]]
                    features.append(vals)
                    ids.append(arr[0])
        except Exception as e:
            print(f"Error: {e}")

    return features, ids
