"""
compare_oversample_ridge.py

Idea 1 of the ridge ablation: oversample the positives in the TRAINING folds
only, then compare the CV F1 for ratio 1 (baseline), 2 and 3.

Setup = 5 folds, seed 1 (build_k_indices), same preprocessing
as the 0.441 run (degree 1, interactions, default max_nan_ratio), lambda 1e-2.
The decision threshold is tuned on the (non-oversampled) training scores of
each fold and applied to the validation fold, exactly as in
cross_validation_ridge().
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
from src.utils.build_k_indices import build_k_indices
from src.utils.cross_validation2 import oversample_positives
from src.utils.metrics import best_threshold, f1_score
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform

SEED = 1
K_FOLD = 5
DEGREE = 1
LAMBDA = 1e-2
RATIOS = [1, 2, 3]  # 1 = baseline, must reproduce ~0.441
DATA_PATH = os.path.join(PROJECT_ROOT, "data")


def run_ratio(y, tx, k_indices, ratio):
    """Per-fold validation F1 of ridge trained with `ratio` copies of each positive."""
    f1s = []
    for k in range(K_FOLD):
        te = k_indices[k]
        tr = k_indices[np.arange(K_FOLD) != k].reshape(-1)
        y_tr, y_te, x_tr, x_te = y[tr], y[te], tx[tr], tx[te]

        # oversample_positives only tests `y == 1`, so it works with the
        # -1/+1 labels as well as with {0, 1}. Training folds only.
        y_os, x_os = oversample_positives(y_tr, x_tr, ratio)
        w, _ = ridge_regression(y_os, x_os, LAMBDA)

        # threshold on the ORIGINAL training scores (true class balance),
        # then applied to the untouched validation fold
        t, _ = best_threshold(x_tr @ w, (y_tr + 1) // 2)
        f1s.append(f1_score((y_te + 1) // 2, ((x_te @ w) >= t).astype(int)))
    return np.mean(f1s), np.std(f1s)


def main():
    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH)
    names = load_feature_names(DATA_PATH)

    params = fit_preprocessing(x_train, names, degree=DEGREE, interactions=True)
    tx = transform(x_train, names, params)
    print("features:", tx.shape[1], "| positive rate:", np.mean(y_train == 1))

    k_indices = build_k_indices(y_train, K_FOLD, SEED)

    rows = []
    for ratio in RATIOS:
        f1_mean, f1_std = run_ratio(y_train, tx, k_indices, ratio)
        label = "baseline (no oversampling)" if ratio == 1 else f"oversample x{ratio}"
        rows.append((label, f1_mean, f1_std))
        print(f"{label:<28} CV F1 = {f1_mean:.4f} +/- {f1_std:.4f}")

    base = rows[0][1]
    print("\n| variant | CV F1 | std | diff vs baseline |")
    print("|---|---|---|---|")
    for label, m, s in rows:
        print(f"| {label} | {m:.4f} | {s:.4f} | {m - base:+.4f} |")


if __name__ == "__main__":
    main()
