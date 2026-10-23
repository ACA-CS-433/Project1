import numpy as np


# the f1 score
def f1_score(y_true, y_pred):
    """

    Args:
        y_true: true prediction
        y_pred: the value predicted

    Returns: the f1 score

    """
    true_positif = np.sum((y_true == 1) & (y_pred == 1))
    false_positif = np.sum((y_true != 1) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred != 1))

    denominator = 2 * true_positif + false_positif + false_negative
    if denominator == 0:
        return 0.0
    return 2 * true_positif / denominator
