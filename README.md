# CS-433 — Project 1: MICHD risk prediction (BRFSS)

- AICrowd: <https://www.aicrowd.com/challenges/epfl-machine-learning-project-1>
- Deadline: **Thursday, October 29, 2026, 4:00 PM**
- Final submission: <http://mlcourse.epfl.ch>

## Team

| Name | GitHub | EPFL email |
|------|--------|------------|
| Adrien Jangal | [@Ryukriss-f](https://github.com/Ryukriss-f)| adrien.jangal@epfl.ch |
| Angéline Bignens | [@AngeB21](https://github.com/AngeB21)  | angeline.bignens@epfl.ch |
| Clara Emmanuelle Chamaa | [@claraemmanuelle](https://github.com/claraemmanuelle) | clara.chamaa@epfl.ch |

AIcrowd team name: `ACA`

## Repository structure

```
.
├── run.py                      # reproduces the best submission (AIcrowd F1 0.444)
├── implementations.py          # the 6 methods required by the project statement
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── helpers.py              # provided by the course (CSV loading + submission)
│   └── utils/                  # preprocessing, cross-validation, metrics, plots
├── logistic_regression/        # logistic regression: run/ (scripts, comparisons)
├── gradient_descent_and_stochastic_gd/     # MSE gd / sgd: run/ and visualization/
├── least_squares_and_ridge_regression/     # least squares and ridge: run/ and visualization/
├── data/                       # x_train.csv, y_train.csv, x_test.csv (ignored by Git)
├── submissions/                # generated CSV files (ignored by Git)
└── report/                     # LaTeX report, 2 pages max
```

## Reproducing the results

Run every command from the project root.

| Result | Command |
|---|---|
| Best submission (AIcrowd F1 0.444) | `python run.py` |
| Ridge with simple preprocessing (test F1 0.421) | `PYTHONPATH=. python least_squares_and_ridge_regression/run/run_ridge.py` |
| Ridge ablation table (5-fold CV, seed 1) | `PYTHONPATH=. python least_squares_and_ridge_regression/run/compare_ridge_combined.py` |
| Validation loss and threshold of ridge | `PYTHONPATH=. python least_squares_and_ridge_regression/run/eval_ridge_loss.py` |
| Least squares on independent columns | `PYTHONPATH=. python least_squares_and_ridge_regression/run/eval_least_squares.py` |
| Least squares on all 614 columns (fails: singular matrix) | `PYTHONPATH=. python least_squares_and_ridge_regression/run/eval_least_squares_raw.py` |
| F1 vs lambda curve | `PYTHONPATH=. python least_squares_and_ridge_regression/visualization/visualization_ridge.py` |

Scripts of the other methods are in `logistic_regression/run` and `gradient_descent_and_stochastic_gd/run`.

## Project constraints

- **Standard Python + NumPy only.** Matplotlib / seaborn are tolerated but
  only for visualization (never in `implementations.py`, `run.py`,...).
- No external dataset.
- Required signatures, not to be modified:
  `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)`,
  `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)`,
  `least_squares(y, tx)`, `ridge_regression(y, tx, lambda_)`,
  `logistic_regression(y, tx, initial_w, max_iters, gamma)`,
  `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)`.
- Each method returns `(w, loss)`: the last `w`, and the corresponding loss. For `ridge_regression` and `reg_logistic_regression`, the loss does not include the penalty term.
- MSE with the `0.5` factor. SGD with a mini-batch of size 1.
- All vectors are 1D, of shape `(D,)` and not `(D, 1)`.
- `np.linalg.solve` is allowed, `np.linalg.lstsq` is not.