"""
run.py

Reproducibility script required by the project statement (Step 4).
"""

import os

import numpy as np

from src.helpers import load_csv_data, create_csv_submission
from implementations import (
    mean_squared_error_gd,
    mean_squared_error_sgd,
    least_squares,
    ridge_regression,
    logistic_regression,
    reg_logistic_regression,
)

SEED = 1
DATA_PATH = "data"
OUTPUT_PATH = os.path.join("submissions", "submission.csv")


def main():
    np.random.seed(SEED)

    # 1. Loading
    # TODO

    # 2. Preprocessing
    # TODO

    # 3. Training the final model
    # TODO

    # 4. Prediction
    # TODO

    # 5. Writing the submission file
    # TODO

    raise NotImplementedError("run.py is not implemented yet.")


if __name__ == "__main__":
    main()
