"""
compare_ridge_combined.py

Comparison table of the ridge regression: each idea is added to a baseline and
the cross-validated F1 is compared (same folds for every variant).

Sections (run all of them, or pick some: `python compare_ridge_combined.py 1 3`):
  1. oversampling of the positives (ratio 1 / 2 / 3), degree 1
  2. extra interaction variables, degree 1
  3. combined, cumulative: degree 2, then + oversample x3, then + extra
     variables

Baseline = degree 1, the 11 interaction variables of preprocessing.py,
default max_nan_ratio, lambda = 1e-2, no oversampling.

Setup = 5 folds, seed 1 (build_k_indices). In each fold the decision threshold
is tuned on the (non-oversampled) training scores and applied to the validation
fold. Oversampling is applied to the training folds only.
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

SEED = 1
K_FOLD = 5
LAMBDA = 1e-3
LAMBDAS_COMBINED = [1e-3, 1e-2]  # degree 2 may prefer a smaller lambda
OVERSAMPLE = 3
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

# copy of the team's current dict (value None = numeric column, int v =
# indicator column == v)
BASE = dict(preprocessing.INTERACTION_VARIABLES)

# Extra interaction variables (BRFSS codes checked on the data):
#   PHYSHLTH  days of bad physical health (88 = none -> 0)
#   EXERANY2  exercise in the past 30 days (1 = yes, 2 = no)
#   _RFBMI5   overweight or obese (1 = no, 2 = yes)
#   EMPLOY1   employment status, 8 = unable to work (indicator)
#   INCOME2   income, 1 (< 10k) ... 8 (>= 75k)
GROUP_A = {"PHYSHLTH": None, "EXERANY2": None, "_RFBMI5": None}
GROUP_B = {"EMPLOY1": 8, "INCOME2": None}
EXTRA = {**GROUP_A, **GROUP_B}

# (label, degree, oversampling ratio, extra interaction variables, lambdas)
SECTIONS = {
    "1": (
        "Idea 1: oversampling (degree 1)",
        [
            ("baseline", 1, 1, {}, [LAMBDA]),
            ("oversample x2", 1, 2, {}, [LAMBDA]),
            ("oversample x3", 1, 3, {}, [LAMBDA]),
        ],
    ),
    "2": (
        "Idea 2: extra interaction variables (degree 1)",
        [
            ("baseline (11 variables)", 1, 1, {}, [LAMBDA]),
            ("+ PHYSHLTH, EXERANY2, _RFBMI5", 1, 1, GROUP_A, [LAMBDA]),
            ("+ EMPLOY1==8, INCOME2", 1, 1, GROUP_B, [LAMBDA]),
            ("+ all 5", 1, 1, EXTRA, [LAMBDA]),
        ],
    ),
    "3": (
        "Combined, cumulative",
        [
            ("0. no interactions (d1)", 1, 1, "NOINT", LAMBDAS_COMBINED),
            ("1. baseline (d1)", 1, 1, {}, LAMBDAS_COMBINED),
            ("2. + degree 2", 2, 1, {}, LAMBDAS_COMBINED),
            (f"3. + oversample x{OVERSAMPLE}", 2, OVERSAMPLE, {}, LAMBDAS_COMBINED),
            ("4. + 5 extra variables", 2, OVERSAMPLE, EXTRA, LAMBDAS_COMBINED),
        ],
    ),
}

# only the last base matrix is kept in memory (they can be > 1 GB)
_CACHE = {"key": None, "tx": None}


def get_tx(x_train, names, degree, extra):
    """Preprocessed matrix for this degree / extra variables (cached).
    extra = "NOINT" means no interaction terms at all."""
    no_int = extra == "NOINT"
    key = (degree, "NOINT" if no_int else tuple(sorted(extra)))
    if _CACHE["key"] != key:
        _CACHE["tx"] = None  # free the memory before rebuilding
        try:
            preprocessing.INTERACTION_VARIABLES = {**BASE, **({} if no_int else extra)}
            params = fit_preprocessing(
                x_train, names, degree=degree, interactions=not no_int
            )
            _CACHE["tx"] = transform(x_train, names, params)
        finally:
            preprocessing.INTERACTION_VARIABLES = BASE
        _CACHE["key"] = key
    return _CACHE["tx"]


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


def print_table(title, rows):
    base = rows[0][3]
    print(f"\n### {title}")
    print("| variant | features | best lambda | CV F1 | std | diff vs baseline |")
    print("|---|---|---|---|---|---|")
    for label, n_features, lam, m, s in rows:
        print(
            f"| {label} | {n_features} | {lam:.0e} | {m:.4f} | {s:.4f} "
            f"| {m - base:+.4f} |"
        )


def main():
    wanted = sys.argv[1:] or list(SECTIONS)
    unknown = [s for s in wanted if s not in SECTIONS]
    if unknown:
        sys.exit(f"Unknown section(s) {unknown}, choose among {list(SECTIONS)}")

    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)
    k_indices = build_k_indices(y_train, K_FOLD, SEED)

    tables = []
    for key in wanted:
        title, variants = SECTIONS[key]
        print(f"\n=== {title} ===")
        rows = []
        for label, degree, ratio, extra, lambdas in variants:
            tx = get_tx(x_train, names, degree, extra)

            best = None
            for lambda_ in lambdas:
                f1_mean, f1_std = cv_f1_ridge(y_train, tx, k_indices, lambda_, ratio)
                print(
                    f"{label:<32} features={tx.shape[1]:<5} lambda={lambda_:.0e} "
                    f"CV F1 = {f1_mean:.4f} +/- {f1_std:.4f}"
                )
                if best is None or f1_mean > best[1]:
                    best = (lambda_, f1_mean, f1_std)
            rows.append((label, tx.shape[1], *best))
        tables.append((title, rows))

    print("\n\n=========== Results tables ===========")
    for title, rows in tables:
        print_table(title, rows)


if __name__ == "__main__":
    main()
