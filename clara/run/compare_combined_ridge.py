"""
compare_combined_ridge.py

Cumulative ladder merging the ridge ideas, to see which steps add up:
  1. baseline            degree 1, 11 interaction variables
  2. + degree 2          squares of the continuous features
  3. + oversample x3     positives repeated in the training folds only
  4. + 5 extra variables PHYSHLTH, EXERANY2, _RFBMI5, EMPLOY1==8, INCOME2
  5. + age features      age^2, age x 5 new variables, age^2 x 10 base variables

Each step keeps everything before it. Every variant is run for each lambda of
LAMBDAS (degree 2 may prefer a different lambda than degree 1) and the best
lambda by CV F1 is reported.

Reuses helpers of compare_interactions_ridge.py and compare_age_ridge.py, which
must be in the same folder. preprocessing.py is not touched.

Setup = the agreed one: 5 folds, seed 1 (build_k_indices); the threshold is
tuned on the (non-oversampled) training scores of each fold.
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
from src.helpers import load_csv_data
from src.utils import preprocessing
from src.utils.build_k_indices import build_k_indices
from src.utils.cross_validation2 import oversample_positives
from src.utils.metrics import best_threshold, f1_score
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform

from compare_age_ridge import AGE, BASE_OTHERS, NEW_VARIABLES, risk_z, standardize
from compare_interactions_ridge import BASE, GROUP_A, GROUP_B

SEED = 1
K_FOLD = 5
LAMBDAS = [1e-3, 1e-2]
OVERSAMPLE = 3
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

# (label, degree, oversampling ratio, extra interaction variables, age features)
VARIANTS = [
    ("1. baseline (d1)", 1, 1, False, False),
    ("2. + degree 2", 2, 1, False, False),
    (f"3. + oversample x{OVERSAMPLE}", 2, OVERSAMPLE, False, False),
    ("4. + 5 extra variables", 2, OVERSAMPLE, True, False),
    ("5. + age features", 2, OVERSAMPLE, True, True),
]


def build_tx(x_train, names, degree, extra):
    """Baseline matrix with the given degree, with or without the 5 extra
    interaction variables (patched in memory, then restored)."""
    variables = {**BASE, **GROUP_A, **GROUP_B} if extra else BASE
    try:
        preprocessing.INTERACTION_VARIABLES = variables
        params = fit_preprocessing(x_train, names, degree=degree, interactions=True)
        return transform(x_train, names, params)
    finally:
        preprocessing.INTERACTION_VARIABLES = BASE


def age_columns(x_clean, names):
    """The 16 age features of compare_age_ridge.py, as one (N, 16) array."""
    age = risk_z(x_clean, names, AGE, None)
    new = [risk_z(x_clean, names, n, v) for n, v in NEW_VARIABLES.items()]
    base = [risk_z(x_clean, names, n, v) for n, v in BASE_OTHERS.items()]
    cols = [standardize(age**2)]
    cols += [standardize(age * z) for z in new]
    cols += [standardize(age**2 * z) for z in base]
    return np.column_stack(cols)


def cv_f1_ridge(y, tx, k_indices, lambda_, ratio):
    """Mean / std over the folds of the validation F1."""
    f1s = []
    for k in range(K_FOLD):
        te = k_indices[k]
        tr = k_indices[np.arange(K_FOLD) != k].reshape(-1)
        y_tr, x_tr = y[tr], tx[tr]

        # oversampling in the training fold only (y == 1 works with -1/+1)
        y_os, x_os = oversample_positives(y_tr, x_tr, ratio)
        w, _ = ridge_regression(y_os, x_os, lambda_)

        # threshold on the original training scores, applied to the validation fold
        t, _ = best_threshold(x_tr @ w, (y_tr + 1) // 2)
        f1s.append(f1_score((y[te] + 1) // 2, ((tx[te] @ w) >= t).astype(int)))
    return np.mean(f1s), np.std(f1s)


def main():
    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)

    rules = preprocessing.special_code_rules(x_train, names)
    x_clean = preprocessing.clean_special_codes(x_train, names, rules)
    age_cols = age_columns(x_clean, names)
    del x_clean

    k_indices = build_k_indices(y_train, K_FOLD, SEED)

    rows = []
    tx, tx_key = None, None
    for label, degree, ratio, extra, age in VARIANTS:
        # the base matrix is only rebuilt when degree / extra variables change
        if (degree, extra) != tx_key:
            tx = None
            tx, tx_key = build_tx(x_train, names, degree, extra), (degree, extra)
        tx_v = np.hstack([tx, age_cols]) if age else tx
        print(f"\n=== {label}: {tx_v.shape[1]} features ===")

        best = None
        for lambda_ in LAMBDAS:
            f1_mean, f1_std = cv_f1_ridge(y_train, tx_v, k_indices, lambda_, ratio)
            print(f"lambda = {lambda_:.0e} | CV F1 = {f1_mean:.4f} +/- {f1_std:.4f}")
            if best is None or f1_mean > best[1]:
                best = (lambda_, f1_mean, f1_std)
        rows.append((label, tx_v.shape[1], *best))
        if age:
            del tx_v

    base_f1 = rows[0][3]
    print("\n| variant | features | best lambda | CV F1 | std | diff vs baseline |")
    print("|---|---|---|---|---|---|")
    for label, n_features, lam, m, s in rows:
        print(
            f"| {label} | {n_features} | {lam:.0e} | {m:.4f} | {s:.4f} | {m - base_f1:+.4f} |"
        )


if __name__ == "__main__":
    main()
