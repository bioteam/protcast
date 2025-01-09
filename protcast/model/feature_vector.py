import os
import sys
import tempfile
import iFeatureOmegaCLI
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
        sys.exit(f"Algorithm '{alg}' is not part of ifeatpro")

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


def get_ifeatureomega_features(alg, seqs):
    """get_ifeatureomega_features
    Returns a list of arrays or "feature vectors" using iFeatureOmega.
    The algorithms are:

    AAC
    EAAC
    CKSAAP_type_1
    CKSAAP_type_2
    DPC_type_1
    DPC_type_2
    DDE
    TPC_type_1
    TPC_type_2
    binary
    binary_6bit
    binary_5bit_type_1
    binary_5bit_type_2
    binary_3bit_type_1
    binary_3bit_type_2
    binary_3bit_type_3
    binary_3bit_type_4
    binary_3bit_type_5
    binary_3bit_type_6
    binary_3bit_type_7
    AESNN3
    GAAC
    EGAAC
    CKSAAGP_type_1
    CKSAAGP_type_2
    GDPC_type_1
    GDPC_type_2
    GTPC_type_1
    GTPC_type_2
    AAIndex
    ZScale
    BLOSUM62
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
    OPF_10bit
    OPF_10bit_type_1
    OPF_7bit_type_1
    OPF_7bit_type_2
    OPF_7bit_type_3
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
    PseKRAAC_type_9
    PseKRAAC_type_10
    PseKRAAC_type_11
    PseKRAAC_type_12
    PseKRAAC_type_13
    PseKRAAC_type_14
    PseKRAAC_type_15
    PseKRAAC_type_16
    KNN

    Parameters
    ----------
    alg: str
        Algorithm name, e.g. 'KNN'
    seqs: dict
        Key is protein id, value is protein sequence

    Returns
    -------
    List of lists of floats
    """
    algs = [
        "AAC",
        "EAAC",
        "CKSAAP_type_1",
        "CKSAAP_type_2",
        "DPC_type_1",
        "DPC_type_2",
        "DDE",
        "TPC_type_1",
        "TPC_type_2",
        "binary",
        "binary_6bit",
        "binary_5bit_type_1",
        "binary_5bit_type_2",
        "binary_3bit_type_1",
        "binary_3bit_type_2",
        "binary_3bit_type_3",
        "binary_3bit_type_4",
        "binary_3bit_type_5",
        "binary_3bit_type_6",
        "binary_3bit_type_7",
        "AESNN3",
        "GAAC",
        "EGAAC",
        "CKSAAGP_type_1",
        "CKSAAGP_type_2",
        "GDPC_type_1",
        "GDPC_type_2",
        "GTPC_type_1",
        "GTPC_type_2",
        "AAIndex",
        "ZScale",
        "BLOSUM62",
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
        "OPF_10bit",
        "OPF_10bit_type_1",
        "OPF_7bit_type_1",
        "OPF_7bit_type_2",
        "OPF_7bit_type_3",
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
        "PseKRAAC_type_9",
        "PseKRAAC_type_10",
        "PseKRAAC_type_11",
        "PseKRAAC_type_12",
        "PseKRAAC_type_13",
        "PseKRAAC_type_14",
        "PseKRAAC_type_15",
        "PseKRAAC_type_16",
        "KNN",
    ]
    if alg not in algs:
        sys.exit(f"Algorithm '{alg}' is not part of iFeatureOmegaCLI")

    ids = list()
    features = list()

    for pid, seq in seqs.items():
        tmpfasta = tempfile.NamedTemporaryFile()
        # Create multi-fasta file from input sequences
        with open(tmpfasta.name, "w") as f:
            f.write(">" + pid + "\n" + str(seq.seq) + "\n")
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
            ids.append(pid)
            features.append(vals.tolist())

    return features, ids
