
import numpy as np

from src.utils.metrics import best_threshold, f1_score


def _swap_blocks(a, i, j, size):
    """Swap the rows a[i:i+size] and a[j:j+size] in place (disjoint blocks)."""
    tmp = a[i : i + size].copy()
    a[i : i + size] = a[j : j + size]
    a[j : j + size] = tmp


def oversample_positives(y, tx, ratio):
    """Training set where every positive sample appears `ratio` times.

    Equivalent to giving a weight `ratio` to the positive class in the loss.
    Creates a copy of tx (about (1 + (ratio - 1) * positive rate) times its size).

    Args:
        y:     ndarray of shape (N,), labels in {0, 1}
        tx:    ndarray of shape (N, D)
        ratio: int >= 1, number of copies of each positive sample

    Returns:
        (y, tx) with the extra copies of the positive samples appended.
    """
    if ratio <= 1:
        return y, tx
    extra = np.tile(np.flatnonzero(y == 1), ratio - 1)
    return np.concatenate([y, y[extra]]), np.concatenate([tx, tx[extra]])


def cross_validate(y, tx, train_fn, predict_fn=None, k_fold=5, return_models=False):
    """Out-of-fold predictions of a model with k-fold cross-validation.

    Args:
        y:             ndarray of shape (N,), shuffled labels in {0, 1}
        tx:            ndarray of shape (N, D), shuffled features (modified
                       during the call, restored before returning)
        train_fn:      function (y, tx) -> w, trains the model
        predict_fn:    function (tx, w) -> scores, defaults to tx @ w
        k_fold:        number of folds
        return_models: also return the k weight vectors (e.g. to average
                       the predictions of the fold models on the test set)

    Returns:
        (scores, labels, fold_ids): out-of-fold scores, the corresponding
            labels and the fold index of each sample, each of shape (k * m,)
            with m = N // k. The N - k * m first rows are always in the
            training set.
        If return_models: (scores, labels, fold_ids, ws), ws a list of the
            k weight vectors.
    """
    if predict_fn is None:
        predict_fn = lambda tx_, w: tx_ @ w

    n = y.shape[0]
    m = n // k_fold
    last = n - m
    scores, labels, fold_ids, ws = [], [], [], []
    for k in range(k_fold):
        # fold k = rows [n - (k + 1) m, n - k m), moved to the end of the matrix
        start = n - (k + 1) * m
        if k > 0:
            _swap_blocks(tx, start, last, m)
            _swap_blocks(y, start, last, m)
        try:
            w = train_fn(y[:last], tx[:last])
            scores.append(predict_fn(tx[last:], w))
            labels.append(y[last:].copy())
            fold_ids.append(np.full(m, k))
            ws.append(w)
        finally:
            if k > 0:
                _swap_blocks(tx, start, last, m)
                _swap_blocks(y, start, last, m)
    out = (np.concatenate(scores), np.concatenate(labels), np.concatenate(fold_ids))
    return out + (ws,) if return_models else out


def cv_f1(scores, labels, fold_ids):
    """F1 of out-of-fold predictions with a single threshold for all folds.

    Args:
        scores, labels, fold_ids: output of cross_validate()

    Returns:
        (threshold, f1_mean, f1_std): threshold maximizing the F1 on all the
            out-of-fold predictions, and the mean / std of the per-fold F1
            with this threshold.
    """
    threshold, _ = best_threshold(scores, labels)
    f1s = [
        f1_score(labels[fold_ids == k], (scores[fold_ids == k] >= threshold).astype(int))
        for k in np.unique(fold_ids)
    ]
    return threshold, float(np.mean(f1s)), float(np.std(f1s))
