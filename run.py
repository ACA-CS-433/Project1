"""
run.py

Reproducibility script required by the project statement (Step 4).
"""

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
from src.utils.preprocessing import load_feature_names, fit_preprocessing, transform

SEED = 1

MAX_NAN_RATIO = 0.7
DEGREE = 2
INTERACTIONS = True

DATA_PATH = "data"
OUTPUT_PATH = os.path.join("submissions", "submission.csv")

# This file uses mse_gd and get the value of
# max iters = 750 | gamma = 0.01830206 | validation loss = 0.13053418 | f1 score = 0.4310 | threshold = -0.5753


def main():
    np.random.seed(SEED)

    print("Processing data...")
    # 1. Loading
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)

    print("Training data shape: ", x_train.shape)
    print("Test data shape: ", x_test.shape)
    print("Labels shape: ", y_train.shape)

    # Shuffle once: the cross-validation folds are contiguous blocks of rows
    idx = np.random.permutation(y_train.shape[0])
    x_train, y_train = x_train[idx], y_train[idx]

    # 2. Preprocessing
    names = load_feature_names(DATA_PATH)
    params = fit_preprocessing(
        x_train,
        names,
        max_nan_ratio=MAX_NAN_RATIO,
        degree=DEGREE,
        interactions=INTERACTIONS,
    )
    tx_train = transform(x_train, names, params)
    tx_test = transform(x_test, names, params)
    del x_train, x_test
    print("Number of features after preprocessing: ", tx_train.shape[1])

    # 3. Training the final model
    initial_w = np.zeros(tx_train.shape[1])

    # best gradient descent parameters
    # max iters =      750 | gamma = 0.01830206 | validation loss = 0.13053418 | f1 score = 0.4310 | threshold = -0.5753
    max_iters = 750
    gamma = 0.01830206
    best_thd = -0.5753

    w, loss = mean_squared_error_gd(y_train, tx_train, initial_w, max_iters, gamma)
    w[0] -= best_thd

    print("Training loss:", loss)

    # 4. Prediction
    y_pred = tx_test.dot(w)
    # required by the submission format
    y_pred = np.where(y_pred >= 0, 1, -1)

    # 5. Writing the submission file
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()
