import numpy as np


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
    return 2 * true_positif / (2 * true_positif + false_positif + false_negative)
