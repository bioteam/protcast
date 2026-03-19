from __future__ import annotations

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


@typechecked
def calculate_fmax(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple:
    """Calculate Fmax — the CAFA-standard evaluation metric.

    Fmax is the maximum protein-centric F1 score across all decision thresholds.
    At each threshold t, for each protein i:
      - precision_i(t) = |predicted_i ∩ true_i| / |predicted_i|
      - recall_i(t)    = |predicted_i ∩ true_i| / |true_i|
    These are averaged across all proteins, then combined into F1.
    Fmax = max over t of F1(t).

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth multi-hot labels, shape (num_proteins, num_go_terms).
    y_pred : np.ndarray
        Predicted probabilities (sigmoid output), same shape as y_true.
    thresholds : np.ndarray or None
        Thresholds to evaluate. Defaults to np.arange(0.01, 1.0, 0.01).

    Returns
    -------
    tuple
        (fmax, best_threshold) where fmax is the maximum F1 and
        best_threshold is the threshold that achieved it.
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_f1 = 0.0
    best_t = 0.0

    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(np.float32)

        # Per-protein true positives, predicted positives, actual positives
        tp = (y_pred_binary * y_true).sum(axis=1)
        pred_pos = y_pred_binary.sum(axis=1)
        true_pos = y_true.sum(axis=1)

        # Per-protein precision and recall (avoid division by zero)
        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.where(pred_pos > 0, tp / pred_pos, 0.0)
            recall = np.where(true_pos > 0, tp / true_pos, 0.0)

        # Only average over proteins that have at least one true annotation
        has_annotations = true_pos > 0
        if has_annotations.sum() == 0:
            continue

        avg_precision = np.where(
            pred_pos > 0, precision, 0.0
        )[has_annotations].mean()
        avg_recall = recall[has_annotations].mean()

        if avg_precision + avg_recall > 0:
            f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            f1 = 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    return best_f1, best_t


@typechecked
def calculate_smin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple:
    """Calculate Smin — the minimum semantic distance (CAFA metric).

    Simplified version without IC weights (which require the GO DAG).
    Each GO term contributes equally.

    Smin = min over t of sqrt(ru(t)^2 + mi(t)^2)

    Where:
      - ru(t) = avg missed terms per protein (remaining uncertainty)
      - mi(t) = avg wrong terms per protein (misinformation)

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth multi-hot labels, shape (num_proteins, num_go_terms).
    y_pred : np.ndarray
        Predicted probabilities (sigmoid output), same shape as y_true.
    thresholds : np.ndarray or None
        Thresholds to evaluate.

    Returns
    -------
    tuple
        (smin, best_threshold)
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    best_s = float("inf")
    best_t = 0.0

    for t in thresholds:
        y_pred_binary = (y_pred >= t).astype(np.float32)

        tp = (y_pred_binary * y_true).sum(axis=1)
        fn = y_true.sum(axis=1) - tp     # missed terms
        fp = y_pred_binary.sum(axis=1) - tp  # wrong terms

        ru = fn.mean()
        mi = fp.mean()
        s = np.sqrt(ru**2 + mi**2)

        if s < best_s:
            best_s = s
            best_t = float(t)

    return float(best_s), best_t


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
