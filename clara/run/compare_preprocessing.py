"""Ridge with the rich preprocessing vs the simple one (CV F1)."""

import numpy as np

from src.helpers import load_csv_data
from src.utils.cross_validation import cross_validation_ridge_demo
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform

SEED = 21
K_FOLD = 4
LAMBDAS = np.logspace(-8, 0, 9)


def main():
    x_train, x_test, y_train, _, _ = load_csv_data("data")
    names = load_feature_names("data")

    params = fit_preprocessing(x_train, names, degree=1, interactions=True)
    tx = transform(x_train, names, params)
    print("rich preprocessing, features:", tx.shape[1])

    res = cross_validation_ridge_demo(y_train, tx, K_FOLD, LAMBDAS, SEED)
    print("best lambda = %.3e | best CV F1 = %.4f" % (res[0], res[1]))


if __name__ == "__main__":
    main()
