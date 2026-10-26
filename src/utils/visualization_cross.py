"""Code in this file has been written with the help of the ML labs code & LLM"""

import matplotlib.pyplot as plt
import numpy as np


def _positive_only(lambdas, *arrays):
    """Keep only lambda > 0 (lambda = 0, i.e. least squares, cannot be drawn on a log axis)."""
    lambdas = np.asarray(lambdas, dtype=float)
    mask = lambdas > 0
    return (lambdas[mask], *[np.asarray(a)[mask] for a in arrays])


def plot_f1_vs_lambda(lambdas, f1_mean, f1_std=None, best_lambda=None):
    """Plot the mean validation F1 score as a function of lambda (ridge).

    Args:
        lambdas:     array of ridge parameters
        f1_mean:     mean validation F1 over the folds, one value per lambda
        f1_std:      optional, std of the F1 over the folds (drawn as a shaded band)
        best_lambda: optional, the selected lambda (drawn as a vertical line)
    """
    if f1_std is None:
        f1_std = np.zeros_like(f1_mean)
    lam, mean, std = _positive_only(lambdas, f1_mean, f1_std)

    plt.figure()
    plt.plot(lam, mean, marker="o", label="validation F1 (mean over folds)")
    plt.fill_between(lam, mean - std, mean + std, alpha=0.2, label="± 1 std")

    if best_lambda is not None and best_lambda > 0:
        plt.axvline(
            best_lambda,
            color="red",
            linestyle="--",
            label=f"best lambda = {best_lambda:.1e}",
        )

    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("F1 score")
    plt.title("Ridge regression: validation F1 vs lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/f1_vs_lambda.png", dpi=150)
    plt.close()


def plot_loss_vs_lambda(lambdas, loss_tr, loss_te, best_lambda=None):
    """Plot training and validation loss (MSE) as a function of lambda.

    Useful to see overfitting (small lambda: train << validation)
    and underfitting (large lambda: both losses increase).
    """
    lam, tr, te = _positive_only(lambdas, loss_tr, loss_te)

    plt.figure()
    plt.plot(lam, tr, marker="o", label="training loss")
    plt.plot(lam, te, marker="o", label="validation loss")

    if best_lambda is not None and best_lambda > 0:
        plt.axvline(
            best_lambda,
            color="red",
            linestyle="--",
            label=f"best lambda = {best_lambda:.1e}",
        )

    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("MSE loss")
    plt.title("Ridge regression: training and validation loss vs lambda")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/loss_vs_lambda.png", dpi=150)
    plt.close()


def plot_ridge_summary(lambdas, f1_mean, f1_std, loss_tr, loss_te, best_lambda=None):
    """Side-by-side figure: F1 vs lambda (left) and losses vs lambda (right).

    Convenient for the report (one figure, two panels).
    """
    lam, mean, std, tr, te = _positive_only(lambdas, f1_mean, f1_std, loss_tr, loss_te)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(lam, mean, marker="o")
    ax1.fill_between(lam, mean - std, mean + std, alpha=0.2)
    ax1.set_ylabel("validation F1")
    ax1.set_title("F1 vs lambda")

    ax2.plot(lam, tr, marker="o", label="training")
    ax2.plot(lam, te, marker="o", label="validation")
    ax2.set_ylabel("MSE loss")
    ax2.set_title("Loss vs lambda")
    ax2.legend()

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.grid(True, alpha=0.3)
        if best_lambda is not None and best_lambda > 0:
            ax.axvline(best_lambda, color="red", linestyle="--")

    fig.suptitle("Ridge regression: hyperparameter search")
    fig.tight_layout()
    plt.savefig("figures/ridge_summary.png", dpi=150)
    plt.close()
