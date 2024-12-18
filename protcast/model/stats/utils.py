import numpy as np
from sklearn.metrics import (
    f1_score,
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


"""     
F1 Score

The F1 score is calculated using the following formula:

F1 = 2 * (Precision * Recall) / (Precision + Recall)

Where:

Precision is the number of true positives divided by the total number of predicted positives (true positives + false positives). It measures how many of the predicted positive cases were actually positive.   
Recall is the number of true positives divided by the total number of actual positives (true positives + false negatives). It measures how many of the actual positive cases were correctly predicted.   
Alternatively, the F1 score can be expressed in terms of true positives (TP), false positives (FP), and false negatives (FN):   

F1 = 2 * TP / (2 * TP + FP + FN)

Fmax

There isn't a single, direct formula to calculate Fmax. Instead, it's determined algorithmically:

Calculate Precision and Recall for various thresholds: For a given classifier, you can vary the classification threshold (e.g., the probability above which an instance is classified as positive). At each threshold, calculate the precision and recall.
Calculate F1 Score for each threshold: Using the precision and recall values obtained in the previous step, calculate the F1 score for each threshold using the F1 score formula.   
Find the Maximum F1 Score: The Fmax is the highest F1 score among all the calculated F1 scores across the different thresholds.
In essence, finding Fmax involves an optimization process where you search for the threshold that maximizes the F1 score.

Key Points:

F1 score is a single value calculated for a specific threshold.   
Fmax is the maximum F1 score achievable across all possible thresholds.
Calculating Fmax involves an iterative process of calculating F1 scores at different thresholds.
"""
