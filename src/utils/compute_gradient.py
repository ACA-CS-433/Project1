"""Code in this file has been written with the help of the ML labs code"""


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y:      a numpy array of shape (N, )
        tx:     a numpy array of shape (N,2)
        w:      a numpy array of shape (2, ). The vector of model parameters.

    Returns:
        grad:   a numpy array of shape (2, ), containing the gradient of the loss at w.
        err:    a numpy array of shape (N, ), containing the prediction errors.
    """
    N = len(y)
    err = y - tx.dot(w)
    grad = tx.T.dot(err) * (-1 / N)

    return grad, err
