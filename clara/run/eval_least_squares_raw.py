"""Plain least squares (normal equations) on ALL columns, no fallback.
Same setup as the ridge row: 16 variables, degree 2, oversampling x3,
5 folds, seed 1."""

import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from implementations import least_squares
from src.helpers import load_csv_data
from src.utils import preprocessing
from src.utils.build_k_indices import build_k_indices
from src.utils.cross_validation2 import oversample_positives
from src.utils.metrics import best_threshold, f1_score
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform

SEED = 1
K_FOLD = 5
DEGREE = 2
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
    f1s = []
    for k in range(K_FOLD):
        te = k_indices[k]
        tr = k_indices[np.arange(K_FOLD) != k].reshape(-1)
        y_os, x_os = oversample_positives(y_train[tr], tx[tr], OVERSAMPLE)
        try:
            w, _ = least_squares(y_os, x_os)  # no fallback
        except np.linalg.LinAlgError:
            print(f"fold {k}: LinAlgError (singular matrix), least squares fails")
            continue
        t, f1_tr = best_threshold(tx[tr] @ w, y01[tr])
        f1_te = f1_score(y01[te], (tx[te] @ w >= t).astype(int))
        print(
            f"fold {k}: max|w| = {np.abs(w).max():.1f} | "
            f"train F1 = {f1_tr:.4f} | validation F1 = {f1_te:.4f}"
        )
        f1s.append(f1_te)
    if f1s:
        print(f"real least squares: {len(f1s)}/{K_FOLD} folds ran, "
              f"mean F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")


if __name__ == "__main__":
    main()