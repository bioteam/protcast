import numpy as np
from typeguard import typechecked


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
