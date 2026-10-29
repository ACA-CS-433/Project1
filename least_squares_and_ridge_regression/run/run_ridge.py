"""
run_ridge_combined.py
Code in this file has been written with the help of the ML labs code & LLM

Ridge, best step of the combined ladder (compare_ridge_combined.py, section 3, step 4):
  - degree 2, interactions=True
  - 5 extra interaction variables (PHYSHLTH, EXERANY2, _RFBMI5,
    EMPLOY1==8, INCOME2), added in memory: preprocessing.py is not edited
  - positives oversampled x3 in the training data
  - lambda = 1e-3
  - decision threshold tuned on F1 on the (non-oversampled) training scores
CV F1 (5 folds, seed 1) = 0.4311, baseline 0.4295.
"""

import os
import sys

import numpy as np

# go up 2 levels (same convention as the other run scripts)
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from implementations import ridge_regression
from src.helpers import load_csv_data, create_csv_submission
from src.utils import preprocessing
from src.utils.cross_validation2 import oversample_positives
from src.utils.metrics import best_threshold
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform

SEED = 1
DEGREE = 2
LAMBDA = 1e-3
OVERSAMPLE = 3

# extra interaction variables (None = numeric column, int v = indicator == v)
EXTRA_VARIABLES = {
    "PHYSHLTH": None,
    "EXERANY2": None,
    "_RFBMI5": None,
    "EMPLOY1": 8,
    "INCOME2": None,
}

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "submissions", "submission_ridge_combined.csv")


def main():
    np.random.seed(SEED)

    print("Processing data...")
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)

    # shared dict + extra variables, patched in memory only
    base_variables = dict(preprocessing.INTERACTION_VARIABLES)
    preprocessing.INTERACTION_VARIABLES = {**base_variables, **EXTRA_VARIABLES}
    try:
        params = fit_preprocessing(x_train, names, degree=DEGREE, interactions=True)
        tx_train = transform(x_train, names, params)
        tx_test = transform(x_test, names, params)
    finally:
        preprocessing.INTERACTION_VARIABLES = base_variables
    del x_train, x_test
    print("Number of features after preprocessing:", tx_train.shape[1])

    # training: oversampling of the positives (y == 1 works with -1/+1 labels)
    y_os, tx_os = oversample_positives(y_train, tx_train, OVERSAMPLE)
    w, loss = ridge_regression(y_os, tx_os, LAMBDA)
    del y_os, tx_os
    print("Training loss:", loss)

    # threshold on the original (true class balance) training scores
    t, f1_train = best_threshold(tx_train @ w, (y_train + 1) // 2)
    print(f"Threshold = {t:.4f} | training F1 = {f1_train:.4f}")

    y_pred = np.where(tx_test @ w >= t, 1, -1)
    print("Predicted positive rate on test:", np.mean(y_pred == 1))

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print("Submission written to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
