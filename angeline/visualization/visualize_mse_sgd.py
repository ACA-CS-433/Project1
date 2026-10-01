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
from src.utils.visualization import (
    plot_loss_vs_gamma,
    plot_loss_vs_max_iters,
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
GAMMAS = np.logspace(-7, -5, 8)
MAX_ITERS = [500000, 750000, 1000000, 1500000, 2000000]


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

    # 3. Cross validation
    best_gamma, best_max_iters, best_loss, loss_tr, loss_te = cross_validation_mse_demo(
        y_train, tx_train, K_FOLD, MAX_ITERS, GAMMAS, SEED, mean_squared_error_sgd
    )

    # print best parameters
    print()
    print("=" * 60)
    print("best mse sgd parameters")
    print("=" * 60)

    print("best_gamma = %.8f" % best_gamma)
    print("best_max_iters = %d" % best_max_iters)
    print("best_loss = %.8f" % best_loss)

    print("=" * 60)

    # Visualization

    plot_loss_vs_gamma(GAMMAS, MAX_ITERS, loss_te)
    plot_loss_vs_max_iters(MAX_ITERS, GAMMAS, loss_te)
    plot_heatmap(MAX_ITERS, GAMMAS, loss_te)


if __name__ == "__main__":
    main()
