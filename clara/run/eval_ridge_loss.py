"""Validation MSE, threshold and F1 of the final ridge configuration
(614 columns, oversampling x3, lambda = 1e-3), 5 folds, seed 1."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from implementations import ridge_regression
from src.helpers import load_csv_data
from src.utils import preprocessing
from src.utils.build_k_indices import build_k_indices
from src.utils.calculate_mse import calculate_mse
from src.utils.cross_validation2 import oversample_positives
from src.utils.metrics import best_threshold, f1_score
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform

SEED = 1
K_FOLD = 5
DEGREE = 2
LAMBDA = 1e-3
OVERSAMPLE = 3
EXTRA_VARIABLES = {
    "PHYSHLTH": None,
    "EXERANY2": None,
    "_RFBMI5": None,
    "EMPLOY1": 8,
    "INCOME2": None,
}
DATA_PATH = os.path.join(PROJECT_ROOT, "data")


def main():
    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)

    base = dict(preprocessing.INTERACTION_VARIABLES)
    preprocessing.INTERACTION_VARIABLES = {**base, **EXTRA_VARIABLES}
    try:
        params = fit_preprocessing(x_train, names, degree=DEGREE, interactions=True)
        tx = transform(x_train, names, params)
    finally:
        preprocessing.INTERACTION_VARIABLES = base
    print("columns:", tx.shape[1])

    y01 = (y_train + 1) // 2
    k_indices = build_k_indices(y_train, K_FOLD, SEED)
    losses, thresholds, f1s = [], [], []
    for k in range(K_FOLD):
        te = k_indices[k]
        tr = k_indices[np.arange(K_FOLD) != k].reshape(-1)
        y_os, x_os = oversample_positives(y_train[tr], tx[tr], OVERSAMPLE)
        w, _ = ridge_regression(y_os, x_os, LAMBDA)

        t, _ = best_threshold(tx[tr] @ w, y01[tr])
        # validation MSE on the real class balance, {-1, 1} labels, factor 1/2
        loss = calculate_mse(y_train[te] - tx[te] @ w)
        f1 = f1_score(y01[te], (tx[te] @ w >= t).astype(int))
        print(f"fold {k}: loss = {loss:.4f} | threshold = {t:.4f} | F1 = {f1:.4f}")
        losses.append(loss)
        thresholds.append(t)
        f1s.append(f1)

    print(
        f"validation loss = {np.mean(losses):.4f} | "
        f"threshold = {np.mean(thresholds):.4f} | "
        f"F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}"
    )


if __name__ == "__main__":
    main()
