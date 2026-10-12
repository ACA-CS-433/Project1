"""
run_model.py

Command-line version of run_logreg.py for any of the 6 models of
implementations.py: preprocessing, grid search with k-fold cross-validation,
F1-based threshold, submission.

Examples (from the project root):
    python -u adrien/run/run_model.py --model reg_logistic --max-iters 1500 \
        --gamma 0.5 --lambda 0 1e-4 1e-3 --interactions
    python -u adrien/run/run_model.py --model ridge --lambda 1e-5 1e-4 1e-3
    python -u adrien/run/run_model.py --model mse_gd --max-iters 500 --gamma 0.01 0.05
    python -u adrien/run/run_model.py --model logistic --sub-sample --max-iters 100

Every option is documented in parse_args() below.
"""

import argparse
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
from src.utils.cross_validation2 import cross_validate, cv_f1, oversample_positives
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
    sigmoid,
)

# model name -> (uses gamma / max_iters, uses lambda_, outputs probabilities)
MODELS = {
    "mse_gd": (True, False, False),
    "mse_sgd": (True, False, False),
    "least_squares": (False, False, False),
    "ridge": (False, True, False),
    "logistic": (True, False, True),
    "reg_logistic": (True, True, True),
}


def parse_args():
    """Command-line options.

    Options that a model does not use are ignored (e.g. --gamma and
    --max-iters for ridge / least_squares, --lambda for mse_gd / mse_sgd /
    logistic / least_squares).
    """
    parser = argparse.ArgumentParser()

    """
    Model to train:
      mse_gd / mse_sgd         linear regression by (stochastic) gradient descent
      least_squares / ridge    normal equations (no iterations)
      logistic / reg_logistic  (regularized) logistic regression by gradient descent
    """
    parser.add_argument("--model", required=True, choices=MODELS.keys())

    """----- Training -----"""

    """
    Number of gradient descent iterations (gradient models only).
    For mse_sgd one iteration uses a single sample, so much larger values
    are needed.
    """
    parser.add_argument("--max-iters", type=int, default=500)
    
    """
    Learning rate(s) of gradient descent; several values = grid search.
    About 0.5 for logistic models (1.0 diverges), much smaller
    (0.01 - 0.05) for mse_gd.
    """
    parser.add_argument("--gamma", type=float, nargs="+", default=[0.5])

    """
    Regularization parameter(s) of ridge / reg_logistic; several values =
    grid search. Must be > 0 for ridge when features are collinear.
    """
    parser.add_argument(
        "--lambda", dest="lambdas", type=float, nargs="+", default=[1e-3]
    )

    """
    Number of copies of each positive sample in the training folds
    (1 = no oversampling). Equivalent to a weight on the positive class;
    the validation folds keep the true class balance.
    """
    parser.add_argument("--oversample", type=int, default=1)

    """
    Number of cross-validation folds, used to select the hyperparameters
    and the decision threshold.
    """
    parser.add_argument("--k-fold", type=int, default=5)

    """
    Retrain one model on the whole training set for the test predictions.
    By default the predictions are the mean of the k fold models of the
    best configuration (no extra training).
    """
    parser.add_argument("--no-ensemble", action="store_true")

    """----- Preprocessing (see src/utils/preprocessing.py) -----"""

    """
    Add the pairwise products of the key cardiovascular risk factors
    (INTERACTION_VARIABLES, 55 features).
    """
    parser.add_argument("--interactions", action="store_true")

    """1 = continuous features as is, 2 = also add their squares."""
    parser.add_argument("--degree", type=int, default=2, choices=[1, 2])

    """
    Columns with a larger fraction of missing values (after cleaning the
    special codes) are removed.
    """
    parser.add_argument("--max-nan-ratio", type=float, default=0.8)

    """
    Categories (and missing-value indicators) present in less than this
    fraction of the samples are not encoded.
    """
    parser.add_argument("--min-freq", type=float, default=0.01)

    """
    Columns with at most this many distinct values are one-hot encoded,
    the others are treated as continuous.
    """
    parser.add_argument("--max-categories", type=int, default=15)

    """
    Continuous values are clipped to the [q, 1 - q] quantiles of the
    training set to limit the effect of outliers.
    """
    parser.add_argument("--clip-quantile", type=float, default=0.01)

    """----- Misc -----"""

    """Random seed (shuffling / folds, sample order of mse_sgd)."""
    parser.add_argument("--seed", type=int, default=1)

    """Use 1/50 of the training set, for quick tests."""
    parser.add_argument("--sub-sample", action="store_true")

    """Path of the submission file (default: submissions/submission_<model>.csv)."""
    parser.add_argument("--output", default=None)

    return parser.parse_args()


def make_trainer(model, gamma, lambda_, max_iters, oversample, initial_w):
    """Function (y, tx) -> w for the given model and hyperparameters."""

    def train(y, tx):
        y, tx = oversample_positives(y, tx, oversample)
        if model == "mse_gd":
            return mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)[0]
        if model == "mse_sgd":
            return mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)[0]
        if model == "least_squares":
            return least_squares(y, tx)[0]
        if model == "ridge":
            return ridge_regression(y, tx, lambda_)[0]
        if model == "logistic":
            return logistic_regression(y, tx, initial_w, max_iters, gamma)[0]
        return reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)[0]

    return train


def main():
    args = parse_args()
    uses_gamma, uses_lambda, probabilistic = MODELS[args.model]
    gammas = args.gamma if uses_gamma else [None]
    lambdas = args.lambdas if uses_lambda else [None]
    output = args.output or os.path.join(
        PROJECT_ROOT, "submissions", f"submission_{args.model}.csv"
    )
    print("Configuration:", vars(args))

    np.random.seed(args.seed)
    data_path = os.path.join(PROJECT_ROOT, "data")

    # 1. Loading
    print("Processing data...")
    x_train, x_test, y_train, _, test_ids = load_csv_data(
        data_path, sub_sample=args.sub_sample
    )
    print("Training data shape: ", x_train.shape)
    print("Test data shape: ", x_test.shape)

    # Shuffle once: the cross-validation folds are contiguous blocks of rows
    idx = np.random.permutation(y_train.shape[0])
    x_train, y_train = x_train[idx], y_train[idx]

    # 2. Preprocessing, fitted on the training set
    names = load_feature_names(data_path)
    params = fit_preprocessing(
        x_train,
        names,
        max_nan_ratio=args.max_nan_ratio,
        max_categories=args.max_categories,
        min_freq=args.min_freq,
        clip_quantile=args.clip_quantile,
        degree=args.degree,
        interactions=args.interactions,
    )
    tx_train = transform(x_train, names, params)
    tx_test = transform(x_test, names, params)
    del x_train, x_test
    print("Number of features after preprocessing: ", tx_train.shape[1])

    # Labels in {0, 1} for every model: for the regression models this is an
    # affine change of {-1, 1}, which gives the same ranking of the samples,
    # hence the same F1 after threshold tuning
    y = (y_train + 1) / 2

    # 3. Grid search with k-fold cross-validation
    if probabilistic:
        predict = lambda tx, w: sigmoid(tx @ w)
    else:
        predict = lambda tx, w: tx @ w
    initial_w = np.zeros(tx_train.shape[1])
    best = None
    for gamma in gammas:
        for lambda_ in lambdas:
            t0 = time.time()
            train = make_trainer(
                args.model, gamma, lambda_, args.max_iters, args.oversample, initial_w
            )
            try:
                scores, labels, folds, ws = cross_validate(
                    y, tx_train, train, predict, args.k_fold, return_models=True
                )
            except np.linalg.LinAlgError:
                print(
                    f"gamma={gamma} lambda={lambda_}: singular matrix "
                    "(collinear features), try --model ridge with a small --lambda"
                )
                continue
            if not np.isfinite(scores).all():
                print(f"gamma={gamma} lambda={lambda_}: diverged, lower --gamma")
                continue
            th, f1_mean, f1_std = cv_f1(scores, labels, folds)
            print(
                f"gamma={gamma} lambda={lambda_} cv_F1={f1_mean:.4f} "
                f"+/- {f1_std:.4f} threshold={th:.4f} ({time.time() - t0:.0f}s)"
            )
            if best is None or f1_mean > best[0]:
                best = (f1_mean, f1_std, gamma, lambda_, th, ws)

    if best is None:
        sys.exit("No configuration could be trained.")
    f1_mean, f1_std, gamma, lambda_, th, ws = best
    print(
        f"Best: gamma={gamma} lambda={lambda_} threshold={th:.4f} "
        f"cv_F1={f1_mean:.4f} +/- {f1_std:.4f}"
    )

    # 4. Final model
    if args.no_ensemble:
        train = make_trainer(
            args.model, gamma, lambda_, args.max_iters, args.oversample, initial_w
        )
        scores_test = predict(tx_test, train(y, tx_train))
    else:
        # average of the k fold models: the threshold was chosen on the
        # out-of-fold predictions of these same models
        scores_test = np.mean([predict(tx_test, w) for w in ws], axis=0)
        print(f"Ensemble of the {len(ws)} fold models")

    # 5. Prediction (required by the submission format: {-1, 1})
    y_pred = np.where(scores_test >= th, 1, -1)
    print("Predicted positive rate on test: ", np.mean(y_pred == 1))

    # 6. Writing the submission file
    os.makedirs(os.path.dirname(output), exist_ok=True)
    create_csv_submission(test_ids, y_pred, output)
    print("Submission written to", output)


if __name__ == "__main__":
    main()
