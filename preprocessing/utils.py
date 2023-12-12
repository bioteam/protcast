import os
import tempfile

from ifeatpro.features import get_feature
import numpy as np
from typeguard import typechecked


@typechecked
def get_protein_feature(
    feature: str, seqs: list[Seq], pids: list[str]
) -> list[list[float]]:
    """get_protein_feature
    Returns a list of arrays or "feature vectors" using ifeatpro

    Parameters
    ----------
    feature: str
        Feature name, e.g. 'ctriad'
    seqs: list
        List of sequence strings
    pids: list
        List of Swisssprot accessions, e.g. 'ARAB_ABC5'

    Returns
    -------
    List of lists of floats
    """
    tmpdir = tempfile.TemporaryDirectory()
    tmpfasta = tempfile.NamedTemporaryFile()
    # Write fasta file
    with open(tmpfasta.name, "w") as f:
        for (pid, seq) in zip(pids, seqs):
            f.write(">" + pid + "\n" + seq + "\n")

    get_feature(tmpfasta.name, feature, tmpdir.name)
    with open(os.path.join(tmpdir.name, feature + ".csv")) as f:
        features = [[float(x) for x in line.split(",")[1:]] for line in f.readlines()]
    return features


@typechecked
def calculate_mean_imbalance_ratio(labels: np.ndarray) -> float:
    """calculate_mean_imbalance_ratio
    Calculates the mean imbalance ratio. Uses imbalance ratio per label (IRLbl).

    Parameters
    ----------
    labels: np.ndarray
        ...

    Returns
    -------
    Float
    """
    # IRLBL numerator
    sum_array = np.count_nonzero(labels, axis=0)
    irlbl_num = sum_array.max()
    n_classes = labels.shape[1]

    ratio_sum = np.sum(irlbl_num / sum_array)
    return ratio_sum / n_classes
