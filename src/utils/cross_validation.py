"""Code in this file has been written with the help of the ML labs code & LLM"""

import numpy as np


from src.utils.build_k_indices import build_k_indices
from src.utils.calculate_mse import calculate_mse
from src.utils.compute_gradient import compute_gradient


def cross_validation_mse(y, x, k_indices, k, max_iters, gamma, method):
    """Perform one fold of K-fold cross-validation using mse gd or mse sgd

    Args:
        y:          shape=(N,)
        x:          shape=(N,D)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        max_iters:     a list
        gamma:      a list
        method:     mean_squared_error_gd or mean_squared_error_sgd

    Returns:
        loss_tr: training loss
        loss_te: validation (test) loss
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

    return float(loss_tr), float(loss_te)


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
        best_gamma:   scalar, value of the best gamma
        best_max_iters: scalar, value of the best max_iters
        best_loss:     scalar, the associated loss for the best gamma/max_iters
        loss_tr: training loss
        loss_te: validation (test) loss
    """

    seed = seed
    max_iters = max_iters
    k_fold = k_fold
    gammas = gammas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    loss_tr = np.zeros((len(max_iters), len(gammas)))
    loss_te = np.zeros((len(max_iters), len(gammas)))

    # cross validation over gammas and max_iters
    for i, max_iter in enumerate(max_iters):
        for j, gamma in enumerate(gammas):
            loss_tr_tmp = []
            loss_te_tmp = []
            for k in range(k_fold):
                loss_trs, loss_tes = cross_validation_mse(
                    y, x, k_indices, k, max_iter, gamma, method
                )
                loss_tr_tmp.append(loss_trs)
                loss_te_tmp.append(loss_tes)
            loss_tr[i, j] = np.mean(loss_tr_tmp)
            loss_te[i, j] = np.mean(loss_te_tmp)

            print(
                f"max iters = {max_iter:8d} | gamma = {gamma:.8f} | validation loss = {loss_te[i, j]:.8f}"
            )

    # Find best combination
    best_i, best_j = np.unravel_index(np.argmin(loss_te), loss_te.shape)
    best_gamma = gammas[best_j]
    best_max_iters = max_iters[best_i]
    best_loss = loss_te[best_i, best_j]

    return best_gamma, best_max_iters, best_loss, loss_tr, loss_te
