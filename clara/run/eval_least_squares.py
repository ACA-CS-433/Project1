"""Out-of-sample F1 of plain least squares (normal equations, no penalty),
on the rich features with linearly dependent columns removed."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from implementations import least_squares
from src.helpers import load_csv_data
from src.utils.build_k_indices import build_k_indices
from src.utils.metrics import best_threshold, f1_score
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform
from src.utils import preprocessing
from src.utils.cross_validation2 import oversample_positives

OVERSAMPLE = 3
EXTRA_VARIABLES = {
    "PHYSHLTH": None,
    "EXERANY2": None,
    "_RFBMI5": None,
    "EMPLOY1": 8,
    "INCOME2": None,
}

SEED = 1
K_FOLD = 5
DEGREE = 2
DATA_PATH = os.path.join(PROJECT_ROOT, "data")


def independent_columns(a, tol=1e-6):
    """Greedy choice of linearly independent columns, using only a = X^T X.
    A column is kept if its part not explained by the kept columns
    is larger than tol (relative to its own norm)."""
    keep = []
    for j in range(a.shape[0]):
        if keep:
            sub = a[np.ix_(keep, keep)]
            coef = np.linalg.solve(sub, a[keep, j])
            resid = a[j, j] - a[keep, j] @ coef
        else:
            resid = a[j, j]
        if resid > tol * a[j, j]:
            keep.append(j)
    return np.array(keep)


def main():
    print("Processing data...")
    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)

    # same 16 interaction variables as the ridge run (patched in memory only)
    base_variables = dict(preprocessing.INTERACTION_VARIABLES)
    preprocessing.INTERACTION_VARIABLES = {**base_variables, **EXTRA_VARIABLES}
    try:
        params = fit_preprocessing(x_train, names, degree=DEGREE, interactions=True)
        tx = transform(x_train, names, params)
    finally:
        preprocessing.INTERACTION_VARIABLES = base_variables

    # chosen once on the features only (no labels, so no leakage)
    keep = independent_columns(tx.T @ tx)
    tx = tx[:, keep]
    a = tx.T @ tx
    print(f"columns kept: {tx.shape[1]}")
    print(f"rank(X^T X) = {np.linalg.matrix_rank(a)} | cond = {np.linalg.cond(a):.2e}")

    y01 = (y_train + 1) // 2
    k_indices = build_k_indices(y_train, K_FOLD, SEED)
    f1s = []
    for k in range(K_FOLD):
        te = k_indices[k]
        tr = k_indices[np.arange(K_FOLD) != k].reshape(-1)

        # oversample the positives in the training fold only
        y_os, x_os = oversample_positives(y_train[tr], tx[tr], OVERSAMPLE)
        w, _ = least_squares(y_os, x_os)  # plain normal equations

        # threshold on the original (non-oversampled) training scores
        t, f1_tr = best_threshold(tx[tr] @ w, y01[tr])
        f1_te = f1_score(y01[te], (tx[te] @ w >= t).astype(int))
        print(f"fold {k}: train F1 = {f1_tr:.4f} | validation F1 = {f1_te:.4f}")
        f1s.append(f1_te)
    print(f"least squares CV F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")


if __name__ == "__main__":
    main()
