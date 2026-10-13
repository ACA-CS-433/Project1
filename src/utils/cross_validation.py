"""Cross-validation for least squares and ridge regression (F1-based selection)."""

import numpy as np

from implementations import least_squares, ridge_regression
from src.utils.compute_gradient import compute_gradient
from src.utils.calculate_mse import calculate_mse
from src.utils.build_k_indices import build_k_indices
from src.utils.f1_score import f1_score  


def best_threshold(y_true, y_score, quantiles=np.linspace(0.5, 0.99, 50)):
    """Find the decision threshold that maximizes F1.

    A threshold t turns a real-valued score into a class:
    score >= t -> +1, otherwise -1. Candidates are quantiles of the scores.

    Returns:
        (best_t, best_f1)
    """
    best_t, best_f1 = None, -1.0
    for t in np.quantile(y_score, quantiles): #t is the threshold 
        f1 = f1_score(y_true, np.where(y_score >= t, 1, -1))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def fit_model(y, x, lambda_):
    """Train least squares (lambda_=0) or ridge; fall back to ridge if singular."""
    if lambda_ == 0:
        try:
            return least_squares(y, x)
        except np.linalg.LinAlgError:
            return ridge_regression(y, x, 1e-8)  # tiny ridge to stabilize
    return ridge_regression(y, x, lambda_)


def cross_validation_ridge(y, x, k_indices, k, lambda_):
    """Perform one fold of K-fold cross-validation. lambda_=0 gives plain least squares.

    Args:
        y:          shape (N,), labels in {-1, +1}
        x:          shape (N, D)
        k_indices:  array from build_k_indices()
        k:          scalar, index of the validation fold
        lambda_:    ridge parameter
       
    Returns:
        loss_tr: training loss
        loss_te: validation (test) loss,
        f1_te:   f1 score
    """
    # get k'th subgroup in test, others in train:
    te_idx = k_indices[k]
    # take all the elements except the one at position k
    tr_idx = k_indices[np.arange(k_indices.shape[0]) != k].reshape(-1)
    y_tr, y_te = y[tr_idx], y[te_idx]
    x_tr, x_te = x[tr_idx], x[te_idx]


    w, loss_tr = fit_model(y_tr, x_tr, lambda_)

    _, error_te = compute_gradient(y_te, x_te, w)
    loss_te = calculate_mse(error_te)

    # threshold chosen on train scores, applied to the validation fold
    score_tr = x_tr @ w
    score_te = x_te @ w

    t, _ = best_threshold(y_tr, score_tr)
    f1_te = f1_score(y_te, np.where(score_te >= t, 1, -1))

    return float(loss_tr), float(loss_te), float(f1_te)


def cross_validation_ridge_demo(y, x, k_fold, lambdas, seed):
    """cross validation over lambdas, selecting the best one by mean validation F1.

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_fold:     integer, the number of folds
        lambdas:    array of ridge parameters
        seed:       random seed

    Returns:
        best_lambda:    scalar, value of the best lambda
        best_f1:        scalar, the associated f1 score for the best lambda
        best_loss:      scalar, the associated loss for the best lambda
        f1_mean:        array of mean f1 scores for each lambda
        f1_std:         array of standard deviations of f1 scores for each lambda
        loss_tr:        training loss
        loss_te:        validation (test) loss
    """
    k_indices = build_k_indices(y, k_fold, seed)
    n = len(lambdas)
    #define arrays to store the losses and f1 scores for each lambda
    loss_tr, loss_te = np.zeros(n), np.zeros(n)
    f1_mean, f1_std = np.zeros(n), np.zeros(n)

    # cross validation over lambdas
    for i, lam in enumerate(lambdas):
        #res contains the training loss, validation loss and f1 score for each fold
        res = np.array([cross_validation_ridge(y, x, k_indices, k, lam) for k in range(k_fold)])
        
        #computes the mean and std of the training loss, validation loss and f1 score over the folds
        loss_tr[i], loss_te[i], f1_mean[i] = res.mean(axis=0)
        f1_std[i] = res[:, 2].std()

        print(
            f"lambda = {lam:.2e} | validation loss = {loss_te[i]:.5f} | "
            f"f1 score = {f1_mean[i]:.4f} +/- {f1_std[i]:.4f}"
        )

    #find index for the best lambda based on the mean f1 score
    best_i = int(np.argmax(f1_mean))

    return lambdas[best_i], f1_mean[best_i], loss_te[best_i], f1_mean, f1_std, loss_tr, loss_te