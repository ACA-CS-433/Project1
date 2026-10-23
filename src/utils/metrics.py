"""Classification metrics and decision threshold selection."""

"""Code in this file has been written with the help of LLM"""

import numpy as np


def f1_score(y_true, y_pred):
    """F1 score of the positive class.

    Args:
        y_true: ndarray of shape (N,), values in {0, 1}
        y_pred: ndarray of shape (N,), values in {0, 1}

    Returns:
        float
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return 2 * tp / (2 * tp + fp + fn) if tp > 0 else 0.0


def best_threshold(scores, y_true, n_thresholds=200):
    """Decision threshold on the scores that maximizes the F1 score.

    The candidate thresholds are quantiles of the scores, so this works for
    probabilities (logistic regression) as well as raw scores (least squares).

    Args:
        scores:       ndarray of shape (N,), higher = more likely positive
        y_true:       ndarray of shape (N,), values in {0, 1}
        n_thresholds: number of candidate thresholds

    Returns:
        (threshold, f1): best threshold (predict 1 if score >= threshold)
                         and the corresponding F1 score.
    """
    # between 50% and 1% of the samples predicted positive
    candidates = np.quantile(scores, np.linspace(0.5, 0.99, n_thresholds))
    f1s = [f1_score(y_true, (scores >= t).astype(int)) for t in candidates]
    i = int(np.argmax(f1s))
    return candidates[i], f1s[i]
