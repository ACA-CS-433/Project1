"""Some helper functions for project 1."""
"""Code in this file has been written with the help of the ML labs code"""

import csv
import numpy as np
import os


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def create_csv_submission(ids, y_pred, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})



def calculate_mse(err):
    """ Calculates the mse.
        Version similar to the one seen in lectures were we use 1/2N.

        Args:
            err:  an error vector of shape (N, )

        Returns:
            a float representing mse.
    """
    N = err.shape[0]
    div = 1/(2*N)

    return div* (err**2).sum()

def compute_gradient(y,tx,w):
    """ Computes the gradient at w.

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
    grad = tx.T.dot(err)*(-1/N)

    return grad,err


def batch_iter(y, tx, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset. Batch size is always set to 1
        Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
        Outputs an iterator which gives mini-batches of 1 matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

        Example of use :
        for minibatch_y, minibatch_tx in batch_iter(y, tx):
            <DO-SOMETHING>
    """
    batch_size = 1
    data_size = len(y)  # Number of data points.
    max_batches = int(data_size / batch_size)  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (data_size - max_batches * batch_size)  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add a random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (start_index + batch_size)  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]