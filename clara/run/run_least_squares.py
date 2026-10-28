"""
run_least_squares.py

Reproducibility script: trains least squares regression with lambda = 0,
tunes the decision threshold on F1, and writes the submission file.
"""

"""Code in this file has been written with the help of the ML labs code & LLM"""

import os
import sys

import numpy as np

# go up 2 levels (same convention as the other run scripts)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.helpers import load_csv_data, create_csv_submission
from implementations import least_squares
from src.utils.metrics import best_threshold
from src.utils.f1_score import f1_score

SEED = 1

# go up 2 levels
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "submissions", "submission_least_squares.csv")


def main():
    np.random.seed(SEED)

    print("Processing data...")
    # 1. Loading
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)

    print("Training data shape: ", x_train.shape)
    print("Test data shape: ", x_test.shape)
    print("Labels shape: ", y_train.shape)

    # 2. Preprocessing (same as the other methods, so results are comparable)
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

    # Add a bias column
    tx_train = np.c_[np.ones(tx_train.shape[0]), tx_train]
    tx_test = np.c_[np.ones(tx_test.shape[0]), tx_test]

    # 3. Training the final model on all the training data
    w, loss = least_squares(y_train, tx_train)
    print("Training loss:", loss)

    # 4. Decision threshold: the scores are real numbers, positives are rare,
    # so we choose the threshold that maximizes F1 on the training scores
    t, f1_train = best_threshold(tx_train @ w, (y_train + 1) // 2)
    print(f"Threshold = {t:.4f} | training F1 = {f1_train:.4f}")

    # 5. Prediction
    scores_test = tx_test @ w
    # required by the submission format
    y_pred = np.where(scores_test >= t, 1, -1)

    # 6. Writing the submission file
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()
