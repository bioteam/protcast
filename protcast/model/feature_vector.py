import os
import sys
import tempfile
import iFeatureOmegaCLI
from ifeatpro.features import get_feature
from Bio.SeqIO import SeqRecord


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
        sys.exit(f"Algorithm '{alg}' is not part of ifeatpro")

    ids = list()
    features = list()

    for pid, seq in seqs.items():
        tmpdir = tempfile.TemporaryDirectory()
        tmpfasta = tempfile.NamedTemporaryFile()
        # Write fasta file
        if isinstance(seq, (SeqRecord)):
            seq = str(seq.seq)
        with open(tmpfasta.name, "w") as f:
            f.write(">" + pid + "\n" + seq + "\n")

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


def get_ifeatureomega_features(alg, seqs):
    """get_ifeatureomega_features
    Returns a list of arrays or "feature vectors" using iFeatureOmega and a dict
    of protein ids and protein sequences. The 49 algorithms are:

    AAC
    CKSAAP_type_1
    CKSAAP_type_2
    DPC_type_1
    DPC_type_2
    DDE
    TPC_type_1
    TPC_type_2
    GAAC
    CKSAAGP_type_1
    CKSAAGP_type_2
    GDPC_type_1
    GDPC_type_2
    GTPC_type_1
    NMBroto
    Moran
    Geary
    CTDC
    CTDT
    CTDD
    CTriad
    KSCTriad
    SOCNumber
    QSOrder
    PAAC
    APAAC
    ASDC
    DistancePair
    AC
    CC
    ACC
    PseKRAAC_type_1
    PseKRAAC_type_2
    PseKRAAC_type_3A
    PseKRAAC_type_3B
    PseKRAAC_type_4
    PseKRAAC_type_5
    PseKRAAC_type_6A
    PseKRAAC_type_6B
    PseKRAAC_type_6C
    PseKRAAC_type_7
    PseKRAAC_type_8
    PseKRAAC_type_10
    PseKRAAC_type_11
    PseKRAAC_type_12
    PseKRAAC_type_13
    PseKRAAC_type_14
    PseKRAAC_type_15
    PseKRAAC_type_16

    Parameters
    ----------
    alg: str
        Algorithm name, e.g. 'ACC'
    seqs: dict
        Key is protein id, value is protein sequence

    Returns
    -------
    List of lists of floats and a list of protein ids
    """
    algs = [
        "AAC",
        "CKSAAP_type_1",
        "CKSAAP_type_2",
        "DPC_type_1",
        "DPC_type_2",
        "DDE",
        "TPC_type_1",
        "TPC_type_2",
        "GAAC",
        "CKSAAGP_type_1",
        "CKSAAGP_type_2",
        "GDPC_type_1",
        "GDPC_type_2",
        "GTPC_type_1",
        "NMBroto",
        "Moran",
        "Geary",
        "CTDC",
        "CTDT",
        "CTDD",
        "CTriad",
        "KSCTriad",
        "SOCNumber",
        "QSOrder",
        "PAAC",
        "APAAC",
        "ASDC",
        "DistancePair",
        "AC",
        "CC",
        "ACC",
        "PseKRAAC_type_1",
        "PseKRAAC_type_2",
        "PseKRAAC_type_3A",
        "PseKRAAC_type_3B",
        "PseKRAAC_type_4",
        "PseKRAAC_type_5",
        "PseKRAAC_type_6A",
        "PseKRAAC_type_6B",
        "PseKRAAC_type_6C",
        "PseKRAAC_type_7",
        "PseKRAAC_type_8",
        "PseKRAAC_type_10",
        "PseKRAAC_type_11",
        "PseKRAAC_type_12",
        "PseKRAAC_type_13",
        "PseKRAAC_type_14",
        "PseKRAAC_type_15",
        "PseKRAAC_type_16",
    ]
    if alg not in algs:
        sys.exit(f"Algorithm '{alg}' is not part of iFeatureOmega")

    pids = list()
    features = list()

    for pid, seq in seqs.items():
        tmpfasta = tempfile.NamedTemporaryFile()
        # Create multi-fasta file from input sequences
        if isinstance(seq, (SeqRecord)):
            seq = str(seq.seq)
        with open(tmpfasta.name, "w") as f:
            f.write(">" + pid + "\n" + seq + "\n")
        # The try is necessary in case the input sequence is too
        # short for a given algorithm
        try:
            protein = iFeatureOmegaCLI.iProtein(tmpfasta.name)
            protein.get_descriptor(alg)
        except Exception as e:
            print(f"Error running {alg} on {pid} sequence: {e}")
            continue
        # proteins.encodings is a dataframe, which is a dict of
        # pandas.core.series.Series objects. A pandas.core.series.Series is a one-dimensional
        # labeled array of values, similar to an array but with labeled elements. The labels
        # in these Series are the column names, like "AAC_A", "AAC_B".
        for pid, vals in protein.encodings.iterrows():
            pids.append(pid)
            features.append(vals.tolist())

    lens = [len(vec) for vec in features]
    print(f"{alg} feature vector length: {set(lens)}")
    return features, pids
