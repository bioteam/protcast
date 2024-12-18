import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
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
    confusion_matrix(y_true, y_pred, normalize="all")
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # Also known as recall
    sens = tp / (tp + fn)
    # Also known as true negative rate
    spec = tn / (tn + fp)
    return f"{sens:.3f}", f"{spec:.3f}"


@typechecked
def calculate_f1_score(y_true: list, y_pred: list) -> float:
    """calculate_f1_score

    Parameters
    ----------
    y_true : list
        _description_
    y_pred : list
        _description_

    Returns
    -------
    f1_score
        float
    """
    return f1_score(y_true, y_pred)


@typechecked
def calculate_fmax_score(y_true: list, y_pred: list) -> float:
    """calculate_fmax_score

    Parameters
    ----------
    y_true : list
        _description_
    y_pred : list
        _description_

    Returns
    -------
    fmax
        float
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    fmax = 2 * precision * recall / (precision + recall)
    return fmax


"""     
Fmax

Fmax, also known as the Maximum F-measure, combines precision and recall:

Fmax = 2 * Precision * Recall / (Precision + Recall)

Where:

Precision: the ratio of true positives to the sum of true positives and false positives (TP / (TP + FP)).
Recall: the ratio of true positives to the sum of true positives and false negatives (TP / (TP + FN)).
Fmax is a non-symmetric metric, meaning it can be affected by variations in precision and recall. 
It's often used when you want to emphasize both accuracy and completeness of predictions.

F1 score

The F1 score, also known as the Harmonic Mean of Precision and Recall, also combines precision and recall:

F1 = 2 * (Precision * Recall) / (Precision + Recall)

The F1 score is also non-symmetric, but it's more sensitive to variations in precision than recall.

The main differences between Fmax and F1 score are:

Fmax: This metric tends to be more sensitive to precision than recall.
F1 score: This metric is more balanced, with a stronger emphasis on precision when precision and recall are similar.

Use Fmax when:
Precision is crucial for your application (e.g., medical diagnosis).
Recall is less important, but still relevant (e.g., spam filtering).

Use the F1 score when:
Both precision and recall are equally important.
You want a more balanced metric that doesn't favor one aspect over the other.
"""
