"""Ridge F1 vs lambda on the final configuration:
degree 2, 16 interaction variables (614 columns), positives oversampled x3
in the training folds only, 5 folds, seed 1."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

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
DEGREE = 2
OVERSAMPLE = 3
LAMBDAS = np.logspace(-8, 0, 9)
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

    means, stds = [], []
    for lam in LAMBDAS:
        f1s = []
        for k in range(K_FOLD):
            te = k_indices[k]
            tr = k_indices[np.arange(K_FOLD) != k].reshape(-1)
            y_os, x_os = oversample_positives(y_train[tr], tx[tr], OVERSAMPLE)
            w, _ = ridge_regression(y_os, x_os, lam)
            t, _ = best_threshold(tx[tr] @ w, y01[tr])
            f1s.append(f1_score(y01[te], (tx[te] @ w >= t).astype(int)))
        means.append(np.mean(f1s))
        stds.append(np.std(f1s))
        print(f"lambda = {lam:.1e} | CV F1 = {means[-1]:.4f} +/- {stds[-1]:.4f}")

    means, stds = np.array(means), np.array(stds)
    best = LAMBDAS[int(np.argmax(means))]
    print(f"best lambda = {best:.1e}")

    os.makedirs("figures", exist_ok=True)
    plt.figure()
    plt.plot(LAMBDAS, means, marker="o", label="validation F1 (mean over folds)")
    plt.fill_between(LAMBDAS, means - stds, means + stds, alpha=0.2, label="± 1 std")
    plt.axvline(best, color="red", linestyle="--", label=f"best lambda = {best:.1e}")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("F1 score")
    plt.title("Ridge: validation F1 vs lambda (final configuration)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/f1_vs_lambda_final.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()