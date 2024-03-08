import os
import tempfile
from ifeatpro.features import get_feature
from typeguard import typechecked


@typechecked
def get_ifeatpro_features(feature: str, seqs: dict[str, str]) -> list[list[float]]:
    """get_ifeatpro_features
    Returns a list of arrays or "feature vectors" using ifeatpro

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
    # Write fasta file
    with open(tmpfasta.name, "w") as f:
        for pid, seq in seqs.items:
            f.write(">" + pid + "\n" + seq + "\n")

    get_feature(tmpfasta.name, feature, tmpdir.name)
    with open(os.path.join(tmpdir.name, feature + ".csv")) as f:
        features = [[float(x) for x in line.split(",")[1:]] for line in f.readlines()]
    return features
