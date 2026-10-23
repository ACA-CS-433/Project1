"""
compare_interactions_ridge.py

Idea 2 of the ridge ablation: add more risk factors to the pairwise
interactions and compare the CV F1.

preprocessing.py is shared with the team, so it is NOT edited: the module
variable INTERACTION_VARIABLES is replaced in memory for each variant
(fit_preprocessing reads it at call time) and restored at the end.

Setup = the agreed one: 5 folds, seed 1 (build_k_indices), degree 1,
lambda 1e-2, default max_nan_ratio, threshold tuned on the training scores of
each fold (as in cross_validation_ridge).
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

# copy of the team's current dict (value None = numeric column, int v =
# indicator column == v)
BASE = dict(preprocessing.INTERACTION_VARIABLES)

# Codes (BRFSS 2015), to be confirmed by check_codes() below:
#   PHYSHLTH  days of bad physical health, 1-30, 88 = none (-> 0), 77/99 missing
#   EXERANY2  exercise in the past 30 days, 1 = yes, 2 = no, 7/9 missing
#   _RFBMI5   overweight or obese, 1 = no, 2 = yes, 9 missing
#   EMPLOY1   1-8 non-ordinal (7 = retired, 8 = unable to work) -> indicator
#   INCOME2   1 (< 10k) ... 8 (>= 75k), 77/99 missing
GROUP_A = {"PHYSHLTH": None, "EXERANY2": None, "_RFBMI5": None}
GROUP_B = {"EMPLOY1": 8, "INCOME2": None}

VARIANTS = [
    ("baseline (11 variables)", {}),
    ("+ PHYSHLTH, EXERANY2, _RFBMI5", GROUP_A),
    ("+ EMPLOY1==8, INCOME2", GROUP_B),
    ("+ all 5", {**GROUP_A, **GROUP_B}),
]


def check_codes(x_train, names, new_variables):
    """Print the raw values, the special-code rule and the missing rate of
    each new variable, to verify the codes on the real data."""
    rules = preprocessing.special_code_rules(x_train, names)
    cleaned = preprocessing.clean_special_codes(x_train, names, rules)
    for name in new_variables:
        if name not in names:
            raise SystemExit(f"{name} is not a column of x_train.csv")
        j = names.index(name)
        raw = x_train[:, j]
        values = np.unique(raw[~np.isnan(raw)])
        shown = ", ".join(f"{v:g}" for v in values[:12])
        if len(values) > 12:
            shown += f", ... ({len(values)} values)"
        missing, zero = rules.get(name, ([], []))
        print(
            f"{name:<10} raw values: {shown}\n"
            f"{'':<10} missing codes {missing} | zero codes {zero} | "
            f"NaN after cleaning: {np.mean(np.isnan(cleaned[:, j])):.1%}"
        )


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

    print("=== Code check ===")
    check_codes(x_train, names, {**GROUP_A, **GROUP_B})

    k_indices = build_k_indices(y_train, K_FOLD, SEED)

    print("\n=== CV ===")
    rows = []
    try:
        for label, extra in VARIANTS:
            preprocessing.INTERACTION_VARIABLES = {**BASE, **extra}
            params = fit_preprocessing(
                x_train, names, degree=DEGREE, interactions=True
            )
            tx = transform(x_train, names, params)
            f1_mean, f1_std = cv_f1_ridge(y_train, tx, k_indices)
            rows.append((label, tx.shape[1], f1_mean, f1_std))
            print(
                f"{label:<32} features={tx.shape[1]:<5} "
                f"CV F1 = {f1_mean:.4f} +/- {f1_std:.4f}"
            )
            del tx
    finally:
        preprocessing.INTERACTION_VARIABLES = BASE

    base = rows[0][2]
    print("\n| variant | features | CV F1 | std | diff vs baseline |")
    print("|---|---|---|---|---|")
    for label, n_features, m, s in rows:
        print(f"| {label} | {n_features} | {m:.4f} | {s:.4f} | {m - base:+.4f} |")


if __name__ == "__main__":
    main()