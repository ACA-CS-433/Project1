"""
implementations.py

The 6 mandatory methods.

Constraints imposed by the project statement:
  - Only NumPy and the standard library are allowed.
  - Each function returns the tuple (w, loss):
        w    : ndarray of shape (D,)  -> LAST weight vector
        loss : float                  -> loss corresponding to this w
  - For ridge_regression and reg_logistic_regression, the returned loss
    does NOT include the penalty term.
  - MSE with the 0.5 factor (consistent with the lecture notes).
  - For SGD: mini-batch of size 1 (a single randomly drawn point).
  - All vectors are 1D arrays of shape (X,), never (X, 1).
"""

import numpy as np


"""Numerically stable sigmoid, applied element-wise.

    Args:
        t: ndarray of any shape

    Returns:
        ndarray of the same shape, values in (0, 1).
"""


def sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))


"""Negative log-likelihood of the logistic model, averaged over the samples.

    Args:
        y:  ndarray of shape (N,), values in {0, 1}
        tx: ndarray of shape (N, D)
        w:  ndarray of shape (D,)

    Returns:
        float: mean over n of [log(1 + exp(x_n^T w)) - y_n * x_n^T w].
"""


def compute_logistic_loss(y, tx, w):
    z = tx @ w
    return np.mean(np.logaddexp(0.0, z) - y * z)


"""Gradient of the (averaged) logistic negative log-likelihood.

    Args:
        y:  ndarray of shape (N,), values in {0, 1}
        tx: ndarray of shape (N, D)
        w:  ndarray of shape (D,)

    Returns:
        ndarray of shape (D,).
"""


def compute_logistic_gradient(y, tx, w):
    return tx.T @ (sigmoid(tx @ w) - y) / y.shape[0]


"""Linear regression using gradient descent.

    Args:
        y:         ndarray of shape (N,)
        tx:        ndarray of shape (N, D)
        initial_w: ndarray of shape (D,), initial weights
        max_iters: int, number of iterations
        gamma:     float, learning rate

    Returns:
        (w, loss): last weight vector and its MSE loss.

    """


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):

    # TODO
    raise NotImplementedError


"""Linear regression using SGD, mini-batch of size 1.

    Args:
        y:         ndarray of shape (N,)
        tx:        ndarray of shape (N, D)
        initial_w: ndarray of shape (D,)
        max_iters: int, number of iterations
        gamma:     float, learning rate

    Returns:
        (w, loss): last w and the MSE loss computed on the WHOLE dataset.
"""


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):

    # TODO
    raise NotImplementedError


"""Least squares using the normal equations.

    Args:
        y:  ndarray of shape (N,)
        tx: ndarray of shape (N, D)

    Returns:
        (w, loss): optimal solution and its MSE loss.
"""


def least_squares(y, tx):
    # TODO
    raise NotImplementedError


"""Ridge regression using the normal equations.

    Args:
        y:       ndarray of shape (N,)
        tx:      ndarray of shape (N, D)
        lambda_: float, regularization parameter

    Returns:
        (w, loss): solution and its MSE loss WITHOUT the penalty term.
"""


def ridge_regression(y, tx, lambda_):

    # TODO
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y:         ndarray of shape (N,), values in {0, 1}
        tx:        ndarray of shape (N, D)
        initial_w: ndarray of shape (D,)
        max_iters: int
        gamma:     float

    Returns:
        (w, loss): last w and its negative log-likelihood.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        w = w - gamma * compute_logistic_gradient(y, tx, w)
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent (y in {0, 1}).

    Args:
        y:         ndarray of shape (N,), values in {0, 1}
        tx:        ndarray of shape (N, D)
        lambda_:   float, regularization parameter
        initial_w: ndarray of shape (D,)
        max_iters: int
        gamma:     float

    Returns:
        (w, loss): last w and its negative log-likelihood without penalty.
    """
    w = np.array(initial_w, dtype=float)
    for _ in range(max_iters):
        grad = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * grad
    loss = compute_logistic_loss(y, tx, w)
    return w, loss
