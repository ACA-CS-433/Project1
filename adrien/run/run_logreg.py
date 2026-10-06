import os
import sys
import time

import numpy as np

# go up 2 levels
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, PROJECT_ROOT)

from src.helpers import load_csv_data, create_csv_submission
from src.utils.preprocessing import load_feature_names, fit_preprocessing, transform
from src.utils.cross_validation import cross_validate, cv_f1
from implementations import reg_logistic_regression, sigmoid

SEED = 1
SUB_SAMPLE = False

DATA_PATH = os.path.join(PROJECT_ROOT, "data")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "submissions", "submission_logreg.csv")

MAX_NAN_RATIO = 0.8
DEGREE = 2
K_FOLD = 5
MAX_ITERS = 500
GAMMAS = [0.2, 0.35, 0.5]
LAMBDAS = [0, 1e-4, 1e-3]


def main():
    np.random.seed(SEED)

    print("Processing data...")
    # 1. Loading
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(
        DATA_PATH, sub_sample=SUB_SAMPLE
    )

    print("Training data shape: ", x_train.shape)
    print("Test data shape: ", x_test.shape)
    print("Positive rate: ", np.mean(y_train == 1))

    # Shuffle once: the cross-validation folds are contiguous blocks of rows
    idx = np.random.permutation(y_train.shape[0])
    x_train, y_train = x_train[idx], y_train[idx]

    # 2. Preprocessing (special codes, one-hot, missing indicators,
    # clipping, squares, standardization, bias) fitted on the training set
    names = load_feature_names(DATA_PATH)
    params = fit_preprocessing(
        x_train, names, max_nan_ratio=MAX_NAN_RATIO, degree=DEGREE
    )
    tx_train = transform(x_train, names, params)
    tx_test = transform(x_test, names, params)
    del x_train, x_test
    print("Number of features after preprocessing: ", tx_train.shape[1])

    # Logistic loss expects labels in {0, 1}
    y = (y_train + 1) / 2

    # 3. Hyperparameter search with k-fold cross-validation
    # (reg_logistic_regression with lambda_ = 0 is logistic_regression)
    initial_w = np.zeros(tx_train.shape[1])
    predict = lambda tx, w: sigmoid(tx @ w)
    best = None
    for gamma in GAMMAS:
        for lambda_ in LAMBDAS:
            t0 = time.time()
            train = lambda y_, tx_: reg_logistic_regression(
                y_, tx_, lambda_, initial_w, MAX_ITERS, gamma
            )[0]
            scores, labels, folds = cross_validate(y, tx_train, train, predict, K_FOLD)
            th, f1_mean, f1_std = cv_f1(scores, labels, folds)
            print(
                f"gamma={gamma:<5} lambda={lambda_:<7} "
                f"cv_F1={f1_mean:.4f} +/- {f1_std:.4f} threshold={th:.3f} "
                f"({time.time() - t0:.0f}s)"
            )
            if best is None or f1_mean > best[0]:
                best = (f1_mean, f1_std, gamma, lambda_, th)

    f1_mean, f1_std, gamma, lambda_, th = best
    print(
        f"Best: gamma={gamma} lambda={lambda_} threshold={th:.3f} "
        f"cv_F1={f1_mean:.4f} +/- {f1_std:.4f}"
    )

    # 4. Training the final model on the whole training set
    w, loss = reg_logistic_regression(y, tx_train, lambda_, initial_w, MAX_ITERS, gamma)
    print("Final training loss:", loss)

    # 5. Prediction (required by the submission format: {-1, 1})
    y_pred = np.where(sigmoid(tx_test @ w) >= th, 1, -1)
    print("Predicted positive rate on test: ", np.mean(y_pred == 1))

    # 6. Writing the submission file
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    create_csv_submission(test_ids, y_pred, OUTPUT_PATH)
    print("Submission written to", OUTPUT_PATH)


if __name__ == "__main__":
    main()
