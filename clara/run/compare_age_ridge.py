"""
compare_age_ridge.py

Idea 3 of the ridge ablation: age interactions on top of the baseline
(degree 1, interactions=True).

_AGE80 is already crossed with the 10 other risk factors of
INTERACTION_VARIABLES in the baseline, so only what is NOT there is tested:
  - squared age,
  - age x the 5 variables of idea 2 (PHYSHLTH, EXERANY2, _RFBMI5,
    EMPLOY1==8, INCOME2),
  - squared age x the 10 other base risk factors.

The extra columns are built here (median-imputed, standardized, like the
risk factors of preprocessing.py) and appended to the baseline matrix, so
preprocessing.py is not touched.

Setup = the agreed one: 5 folds, seed 1 (build_k_indices), lambda 1e-2,
threshold tuned on the training scores of each fold.
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
from src.utils.metrics import best_threshold, f1_score
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform

SEED = 1
K_FOLD = 5
DEGREE = 1
LAMBDA = 1e-2
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

AGE = "_AGE80"
BASE_OTHERS = {k: v for k, v in preprocessing.INTERACTION_VARIABLES.items() if k != AGE}
NEW_VARIABLES = {
    "PHYSHLTH": None,
    "EXERANY2": None,
    "_RFBMI5": None,
    "EMPLOY1": 8,
    "INCOME2": None,
}


def standardize(c):
    return (c - c.mean()) / max(c.std(), 1e-12)


def risk_z(x_clean, names, name, value):
    """Median-imputed standardized column, as in preprocessing._build_features."""
    col = x_clean[:, names.index(name)]
    if value is not None:
        col = np.where(np.isnan(col), np.nan, (col == value).astype(float))
    col = np.where(np.isnan(col), np.nanmedian(col), col)
    return standardize(col)


def cv_f1_ridge(y, tx, k_indices):
    """Mean / std over the folds of the validation F1 (ridge, lambda = LAMBDA)."""
    f1s = []
    for k in range(K_FOLD):
        te = k_indices[k]
        tr = k_indices[np.arange(K_FOLD) != k].reshape(-1)
        w, _ = ridge_regression(y[tr], tx[tr], LAMBDA)
        t, _ = best_threshold(tx[tr] @ w, (y[tr] + 1) // 2)
        f1s.append(f1_score((y[te] + 1) // 2, ((tx[te] @ w) >= t).astype(int)))
    return np.mean(f1s), np.std(f1s)


def main():
    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)

    rules = preprocessing.special_code_rules(x_train, names)
    x_clean = preprocessing.clean_special_codes(x_train, names, rules)
    j = names.index(AGE)
    print(
        f"{AGE}: min {np.nanmin(x_clean[:, j]):g}, max {np.nanmax(x_clean[:, j]):g}, "
        f"NaN {np.mean(np.isnan(x_clean[:, j])):.1%}, special-code rule {rules.get(AGE)}"
    )

    params = fit_preprocessing(x_train, names, degree=DEGREE, interactions=True)
    tx = transform(x_train, names, params)
    print("baseline features:", tx.shape[1])

    age = risk_z(x_clean, names, AGE, None)
    new = [risk_z(x_clean, names, n, v) for n, v in NEW_VARIABLES.items()]
    base = [risk_z(x_clean, names, n, v) for n, v in BASE_OTHERS.items()]

    groups = {
        "age^2": [standardize(age**2)],
        "age x 5 new variables": [standardize(age * z) for z in new],
        "age^2 x 10 base variables": [standardize(age**2 * z) for z in base],
    }
    variants = [("baseline", [])]
    variants += [(f"+ {label}", cols) for label, cols in groups.items()]
    variants.append(("+ all three", [c for cols in groups.values() for c in cols]))

    k_indices = build_k_indices(y_train, K_FOLD, SEED)

    rows = []
    for label, cols in variants:
        tx_v = np.hstack([tx, np.column_stack(cols)]) if cols else tx
        f1_mean, f1_std = cv_f1_ridge(y_train, tx_v, k_indices)
        rows.append((label, tx_v.shape[1], f1_mean, f1_std))
        print(
            f"{label:<32} features={tx_v.shape[1]:<5} "
            f"CV F1 = {f1_mean:.4f} +/- {f1_std:.4f}"
        )
        if cols:
            del tx_v

    base_f1 = rows[0][2]
    print("\n| variant | features | CV F1 | std | diff vs baseline |")
    print("|---|---|---|---|---|")
    for label, n_features, m, s in rows:
        print(f"| {label} | {n_features} | {m:.4f} | {s:.4f} | {m - base_f1:+.4f} |")


if __name__ == "__main__":
    main()
