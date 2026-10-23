"""Ridge with rich preprocessing (interactions), threshold tuned on F1."""

import numpy as np

from src.helpers import load_csv_data, create_csv_submission
from implementations import ridge_regression
from src.utils.metrics import best_threshold
from src.utils.preprocessing import fit_preprocessing, load_feature_names, transform

DEGREE = 2  # change to 2 if degree 2 wins in CV
LAMBDA = 1e-3  # best lambda found by CV
OUTPUT_PATH = "submissions/submission_ridge_rich_d2.csv"


def main():
    x_train, x_test, y_train, _, test_ids = load_csv_data("data")
    names = load_feature_names("data")

    params = fit_preprocessing(x_train, names, degree=DEGREE, interactions=True)
    tx_train = transform(x_train, names, params)
    tx_test = transform(x_test, names, params)

    w, loss = ridge_regression(y_train, tx_train, LAMBDA)
    print("Training loss:", loss)

    t, f1_train = best_threshold(tx_train @ w, (y_train + 1) // 2)
    print(f"Threshold = {t:.4f} | training F1 = {f1_train:.4f}")

    y_pred = np.where(tx_test @ w >= t, 1, -1)
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)


if __name__ == "__main__":
    main()
