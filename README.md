# CS-433 — Project 1: MICHD risk prediction (BRFSS)

- Competition: <https://www.aicrowd.com/challenges/epfl-machine-learning-project-1>
- Deadline: **Thursday, October 29, 2026, 4:00 PM**
- Final submission: <http://mlcourse.epfl.ch>

## Team

| Name | GitHub | EPFL email |
|------|--------|------------|
| Adrien Jangal | @Ryukriss-f | adrien.jangal@epfl.ch |
| ... | @... | ...@epfl.ch |
| ... | @... | ...@epfl.ch |

AIcrowd team name: `...`

## Repository structure

```
.
├── run.py                      # reproduces the best submissiont
├── implementations.py          # the 6 methods required by the project statement
├── README.md
├── .gitignore
├── requirements.txt
├── src/    
│   ├── helpers.py              # provided by the course (CSV loading + submission)
│   ├── utils/
│   └── tests/
├── data/                       # x_train.csv, y_train.csv, x_testcsv (ignored by Git)
├── submissions/
└── report/                     # LaTeX (or Typst) report, 2 pages max
```

## Project constraints

- **Standard Python + NumPy only.** Matplotlib / seaborn are tolerated but
  *only* for visualization (so in `src/eda.py`, never in
  `src/implementations.py`, `run.py` or `src/utils/`).
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