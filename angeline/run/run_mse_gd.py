"""
run_mse_gd.py

Reproducibility script required by the project statement (Step 4).
"""

"""Code in this file has been written with the help of the ML labs code & LLM"""

import os

import numpy as np

from src.helpers import load_csv_data, create_csv_submission
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)

SEED = 1

# go up 2 levels
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "submissions", "submission.csv")


def main():
    np.random.seed(SEED)

    print("Processing data...")
    # 1. Loading
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)

    print("Training data shape: ", x_train.shape)
    print("Test data shape: ", x_test.shape)
    print("Labels shape: ", y_train.shape)

    # 2. Preprocessing
    # Handle missing values
    feature_mean = np.nanmean(x_train, axis=0)

    x_train = np.where(np.isnan(x_train), feature_mean, x_train)
    x_test = np.where(np.isnan(x_test), feature_mean, x_test)
    # Scale the features
    mean_x = np.mean(x_train, axis=0)
    std_x = np.std(x_train, axis=0)

    # Avoiding division by 0
    std_x[std_x == 0] = 1
    tx_train = (x_train - mean_x) / std_x
    tx_test = (x_test - mean_x) / std_x

    # 3. Training the final model
    initial_w = np.zeros(tx_train.shape[1])

    # gradient descent parameters
    max_iters = 1000
    gamma = 0.001

    w, loss = mean_squared_error_gd(y_train, tx_train, initial_w, max_iters, gamma)

    print("Training loss:", loss)

    # 4. Prediction
    y_pred = tx_test.dot(w)
    # required by the submission format
    y_pred = np.where(y_pred >= 0, 1, -1)

    # 5. Writing the submission file
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()
