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
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
    # TODO
    raise NotImplementedError


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
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    
    # TODO
    raise NotImplementedError
