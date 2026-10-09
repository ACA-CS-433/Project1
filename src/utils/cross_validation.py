"""Code in this file has been written with the help of the ML labs code & LLM"""

import numpy as np


from src.utils.build_k_indices import build_k_indices
from src.utils.calculate_mse import calculate_mse
from src.utils.compute_gradient import compute_gradient
from src.utils.f1_score import f1_score


def cross_validation_mse(y, x, k_indices, k, max_iters, gamma, method):
    """Perform one fold of K-fold cross-validation using mse gd or mse sgd

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        max_iters:  a scalar
        gamma:      a scalar
        method:     mean_squared_error_gd or mean_squared_error_sgd

    Returns:
        loss_tr: training loss
        loss_te: validation (test) loss,
        f1s:   f1 score
        threshold: the threshold used for the f1_score

    """

    # get k'th subgroup in test, others in train:
    te_indices = k_indices[k]
    # take all the elements except the one at position k
    tr_indices = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indices = tr_indices.reshape(-1)
    y_te = y[te_indices]
    y_tr = y[tr_indices]
    x_te = x[te_indices]
    x_tr = x[tr_indices]

    initial_w = np.zeros(x_tr.shape[1])

    w, loss_tr = method(y_tr, x_tr, initial_w, max_iters, gamma)

    _, error_te = compute_gradient(y_te, x_te, w)
    loss_te = calculate_mse(error_te)

    # Prediction
    y_pred = x_te.dot(w)
    # therehold depending on the quantiles, because the data are inbalanced
    threshold = np.quantile(y_pred, np.linspace(0.5, 0.99, 50))
    f1s = [f1_score(y_te, np.where(y_pred >= t, 1, -1)) for t in threshold]
    index = int(np.argmax(f1s))

    return float(loss_tr), float(loss_te), float(f1s[index]), float(threshold[index])


def cross_validation_mse_demo(y, x, k_fold, max_iters, gammas, seed, method):
    """cross validation over gamma and max_iters

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_fold:     integer, the number of folds
        max_iters:  a list of integers
        gammas:      a list of floats
        seed:       a given seed
        method:     mean_squared_error_gd or mean_squared_error_sgd
    Returns:
        best_gamma:     scalar, value of the best gamma
        best_max_iters: scalar, value of the best max_iters
        best_loss:      scalar, the associated loss for the best gamma/max_iters
        best_f1         scalar, the associated f1 score for the best gamma/max_iters
        f1_te:          f1_score
        loss_tr:        training loss
        loss_te:        validation (test) loss
    """

    seed = seed
    max_iters = max_iters
    k_fold = k_fold
    gammas = gammas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    shape = (len(max_iters), len(gammas))
    # define lists to store the loss of training data and test data
    loss_tr = np.zeros(shape)
    loss_te = np.zeros(shape)
    f1_te = np.zeros(shape)
    threshold_te = np.zeros(shape)

    # cross validation over gammas and max_iters
    for i, max_iter in enumerate(max_iters):
        for j, gamma in enumerate(gammas):
            tr_tmp = []
            te_tmp = []
            f1_tmp = []
            threshold_tmp = []
            for k in range(k_fold):
                fold_tr, fold_te, f1, threshold = cross_validation_mse(
                    y, x, k_indices, k, max_iter, gamma, method
                )
                tr_tmp.append(fold_tr)
                te_tmp.append(fold_te)
                f1_tmp.append(f1)
                threshold_tmp.append(threshold)
            loss_tr[i, j] = np.mean(tr_tmp)
            loss_te[i, j] = np.mean(te_tmp)
            f1_te[i, j] = np.mean(f1_tmp)
            threshold_te[i, j] = np.mean(threshold_tmp)

            print(
                f"max iters = {max_iter:8d} | gamma = {gamma:.8f} | validation loss = {loss_te[i, j]:.8f} | f1 score = {f1_te[i, j]:.4f} | threshold = {threshold_te[i, j]:.4f}"
            )

    # Find best combination, select by loss score
    best_i, best_j = np.unravel_index(np.argmin(loss_te), loss_te.shape)
    best_gamma = gammas[best_j]
    best_max_iters = max_iters[best_i]
    best_loss = loss_te[best_i, best_j]
    # best combination, for f1 score
    best_k, best_l = np.unravel_index(np.argmax(f1_te), f1_te.shape)
    best_f1 = f1_te[best_k, best_l]

    return best_gamma, best_max_iters, best_loss, best_f1, f1_te, loss_tr, loss_te
