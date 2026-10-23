"""
compare_features.py

Compares preprocessing / training variants for regularized logistic
regression with fixed hyperparameters:
  - optional learning curve (train / validation loss and validation F1 every
    CURVE_STEP iterations) to detect under- / over-fitting,
  - k-fold cross-validated F1.
"""

import os
import sys
import time

import numpy as np

# go up 2 levels
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.helpers import load_csv_data
from src.utils.preprocessing import load_feature_names, fit_preprocessing, transform
from src.utils.cross_validation2 import cross_validate, cv_f1, oversample_positives
from src.utils.metrics import best_threshold
from implementations import reg_logistic_regression, compute_logistic_loss, sigmoid

SEED = 1
SUB_SAMPLE = False
DATA_PATH = os.path.join(PROJECT_ROOT, "data")

GAMMA = 0.5
LAMBDA = 1e-3
# 500 iterations are enough to compare variants (relative differences),
# the final model in run_logreg.py uses more
MAX_ITERS = 500
K_FOLD = 5

LEARNING_CURVE = False
CURVE_ITERS = 1500
CURVE_STEP = 100

# (label, options of fit_preprocessing, oversampling ratio of the positives)
VARIANTS = [
    ("no interactions", {}, 1),
    ("interactions", {"interactions": True}, 1),
    ("interactions + oversample x2", {"interactions": True}, 2),
    ("interactions + oversample x3", {"interactions": True}, 3),
]


def learning_curve(y, tx, initial_w):
    """Train on the first (k-1)/k of the rows, validate on the last 1/k.
    Gradient descent has no state besides w, so restarting it from the last
    w for CURVE_STEP iterations continues the same optimization."""
    last = y.shape[0] - y.shape[0] // K_FOLD
    y_tr, tx_tr, y_va, tx_va = y[:last], tx[:last], y[last:], tx[last:]
    w = initial_w
    for it in range(CURVE_STEP, CURVE_ITERS + 1, CURVE_STEP):
        w, train_loss = reg_logistic_regression(
            y_tr, tx_tr, LAMBDA, w, CURVE_STEP, GAMMA
        )
        val_loss = compute_logistic_loss(y_va, tx_va, w)
        _, val_f1 = best_threshold(sigmoid(tx_va @ w), y_va)
        print(
            f"  iter={it:<5} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_F1={val_f1:.4f}"
        )


def main():
    np.random.seed(SEED)
    x_train, _, y_train, _, _ = load_csv_data(DATA_PATH, sub_sample=SUB_SAMPLE)

    # Shuffle once: the cross-validation folds are contiguous blocks of rows
    idx = np.random.permutation(y_train.shape[0])
    x_train, y = x_train[idx], (y_train[idx] + 1) / 2
    names = load_feature_names(DATA_PATH)

    predict = lambda tx, w: sigmoid(tx @ w)
    results = []
    tx, tx_options = None, None
    for label, options, ratio in VARIANTS:
        t0 = time.time()
        # the feature matrix is only rebuilt when the preprocessing changes
        if options != tx_options:
            tx = None
            params = fit_preprocessing(x_train, names, **options)
            tx, tx_options = transform(x_train, names, params), options
        initial_w = np.zeros(tx.shape[1])
        print(f"\n=== {label}: {tx.shape[1]} features ===")

        if LEARNING_CURVE:
            print(f"Learning curve (gamma={GAMMA}, lambda={LAMBDA}):")
            learning_curve(y, tx, initial_w)

        # oversampling only inside the training folds, the validation folds
        # keep the true class balance
        def train(y_, tx_):
            y_, tx_ = oversample_positives(y_, tx_, ratio)
            return reg_logistic_regression(
                y_, tx_, LAMBDA, initial_w, MAX_ITERS, GAMMA
            )[0]

        scores, labels, folds = cross_validate(y, tx, train, predict, K_FOLD)
        th, f1_mean, f1_std = cv_f1(scores, labels, folds)
        print(
            f"{K_FOLD}-fold CV ({MAX_ITERS} iters): cv_F1={f1_mean:.4f} "
            f"+/- {f1_std:.4f} threshold={th:.3f} ({time.time() - t0:.0f}s)"
        )
        results.append((label, tx.shape[1], f1_mean, f1_std))

    print("\n=== Summary ===")
    for label, n_features, f1_mean, f1_std in results:
        print(
            f"{label:<30} features={n_features:<5} cv_F1={f1_mean:.4f} +/- {f1_std:.4f}"
        )


if __name__ == "__main__":
    main()
