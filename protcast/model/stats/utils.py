import numpy as np
from sklearn.metrics import confusion_matrix
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

@typechecked
def calculate_sensitivity_specificity(y_true: list, y_pred: list) -> tuple:
    """calculate_sensitivity_specificity

    Parameters
    ----------
    y_true : list
        _description_
    y_pred : list
        _description_

    Returns
    -------
    tuple
        sensitivity (float), specificity (float)
    """
    confusion_matrix(y_true, y_pred, normalize='all')
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Also known as recall
    sens = tp / (tp + fn)
    # Also known as true negative rate
    spec = tn / (tn + fp)
    return f"{sens:.3f}", f"{spec:.3f}"
