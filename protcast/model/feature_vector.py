import os
import tempfile
from ifeatpro.features import get_feature


def get_ifeatpro_features(feature, seqs):
    """get_ifeatpro_features
    Returns a list of arrays or "feature vectors" using ifeatpro. ifeatpro
    creates a CSV file given a fasta file.

    Parameters
    ----------
    feature: str
        Feature name, e.g. 'ctriad'
    seqs: dict
        Key is protein id, value is protein sequence

    Returns
    -------
    List of lists of floats
    """
    tmpdir = tempfile.TemporaryDirectory()
    tmpfasta = tempfile.NamedTemporaryFile()
    ids = list()
    features = list()
    # Write fasta file
    with open(tmpfasta.name, "w") as f:
        for pid, seq in seqs.items():
            f.write(">" + pid + "\n" + str(seq.seq) + "\n")

    get_feature(tmpfasta.name, feature, tmpdir.name)
    with open(os.path.join(tmpdir.name, feature + ".csv")) as f:
        for line in f.readlines():
            arr = line.rstrip().split(",")
            # Skip the first column which contains the id
            vals = [float(x) for x in arr[1:]]
            features.append(vals)
            ids.append(arr[0])
    return features, ids
