"""Code in this file has been written with the help of the ML labs code"""


def calculate_mse(err):
    """Calculates the mse.
    Version similar to the one seen in lectures were we use 1/2N.

    Args:
        err:  an error vector of shape (N, )

    Returns:
        a float representing mse.
    """
    N = err.shape[0]
    div = 1 / (2 * N)

    return div * (err**2).sum()
