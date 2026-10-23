"""
visualize_ridgepy

K-cross validation for the hyperparameter of least_squares(): lambda.
lambda = 0 corresponds to plain least squares and is used as a baseline.

This file is only for visualization

Code in this file has been written with the help of the ML labs code & LLM
"""

import os

import numpy as np

from src.helpers import load_csv_data
from src.utils.cross_validation import cross_validation_ridge_demo
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform
from src.utils.visualization_cross import (
    plot_f1_vs_lambda,
    plot_loss_vs_lambda,
    plot_ridge_summary,
)

# go up 2 levels
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data")


# Configuration

SEED = 21
K_FOLD = 4
# lambda = 0 first: least squares baseline (not drawn on the log-scale plots)
LAMBDAS = np.concatenate(([0.0], np.logspace(-8, 0, 15)))


def main():
    np.random.seed(SEED)

    print("Processing data...")
    # 1. Loading
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(DATA_PATH)

    print("Training data shape: ", x_train.shape)
    print("Test data shape: ", x_test.shape)
    print("Labels shape: ", y_train.shape)

    # 2. Preprocessing
    names = load_feature_names(DATA_PATH)
    params = fit_preprocessing(x_train, names, degree=1, interactions=False)
    tx_train = transform(x_train, names, params)
    print(tx_train.shape)  # check the number of feature

    # 3. Cross validation
    best_lambda, best_f1, best_loss, f1_mean, f1_std, loss_tr, loss_te = (
        cross_validation_ridge_demo(y_train, tx_train, K_FOLD, LAMBDAS, SEED)
    )

    # print best parameters
    print()
    print("=" * 60)
    print("best ridge parameters")
    print("=" * 60)

    print("best_f1 = %.6f" % best_f1)
    print("best_lambda = %.3e" % best_lambda)
    print("best_loss = %.6f" % best_loss)
    print("ridge (lambda = 0): f1 = %.6f" % f1_mean[0])

    print("=" * 60)

    # Visualization
    plot_f1_vs_lambda(LAMBDAS, f1_mean, f1_std, best_lambda)
    plot_loss_vs_lambda(LAMBDAS, loss_tr, loss_te, best_lambda)
    plot_ridge_summary(LAMBDAS, f1_mean, f1_std, loss_tr, loss_te, best_lambda)


if __name__ == "__main__":
    main()