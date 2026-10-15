"""
visualize mse_sgd

K-cross validation for the 2 hyperparameters of mse_sgd(), gamma and max iters

This file is only for visualization

Code in this file has been written with the help of the ML labs code & LLM
"""

import os

import numpy as np

from implementations import mean_squared_error_sgd
from src.helpers import load_csv_data
from src.utils.cross_validation import cross_validation_mse_demo
from src.utils.preprocessing import load_feature_names, fit_preprocessing, transform
from src.utils.visualization import (
    plot_vs_gamma,
    plot_vs_max_iters,
    plot_heatmap,
)

# go up 2 levels
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
# you need to create a figures folder in order to be able to store them somewhere
FIG_PATH = os.path.join(PROJECT_ROOT, "figures")


# Configuration

SEED = 21
K_FOLD = 4
GAMMAS = np.logspace(-5.4, -4.1, 6)
MAX_ITERS = [50000, 100000, 200000, 500000, 750000]

MAX_NAN_RATIO = 0.7
DEGREE = 2
INTERACTIONS = True


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

    # 3. Cross validation
    (
        best_gamma,
        best_max_iters,
        best_loss,
        best_f1,
        best_thd,
        f1_te,
        loss_tr,
        loss_te,
    ) = cross_validation_mse_demo(
        y_train, tx_train, K_FOLD, MAX_ITERS, GAMMAS, SEED, mean_squared_error_sgd
    )

    # print best parameters
    print()
    print("=" * 60)
    print("best mse sgd parameters")
    print("=" * 60)

    print("best_f1 = %.6f" % best_f1)
    print("best_gamma = %.8f" % best_gamma)
    print("best_max_iters = %d" % best_max_iters)
    print("best_loss = %.8f" % best_loss)
    print("best threshold = %.4f" % best_thd)

    print("=" * 60)

    # Visualization

    plot_vs_gamma(GAMMAS, MAX_ITERS, loss_te, "Loss")
    plot_vs_max_iters(MAX_ITERS, GAMMAS, loss_te, "Loss")
    plot_heatmap(MAX_ITERS, GAMMAS, loss_te, "Loss")
    plot_vs_gamma(GAMMAS, MAX_ITERS, f1_te, "f1")
    plot_vs_max_iters(MAX_ITERS, GAMMAS, f1_te, "f1")
    plot_heatmap(MAX_ITERS, GAMMAS, f1_te, "f1")


if __name__ == "__main__":
    main()
