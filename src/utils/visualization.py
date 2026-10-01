"""Code in this file has been written with the help of the ML labs code & LLM"""

import matplotlib.pyplot as plt
import numpy as np


def plot_loss_vs_gamma(gammas, max_iters, loss_te):
    """Plots validation loss as a function gamma"""
    for i, max_iter in enumerate(max_iters):
        plt.plot(gammas, loss_te[i, :], marker="o", label="max_iters = %d" % max_iter)

    plt.xscale("log")

    plt.xlabel("gamma")
    plt.ylabel("Validation loss")

    plt.title("MSE: Loss vs Gamma")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_loss_vs_max_iters(max_iters, gammas, loss_te):
    """Plots validation loss as a function max_iters"""
    for j, gamma in enumerate(gammas):
        plt.plot(max_iters, loss_te[:, j], marker="o", label="gamma = %.8f" % gamma)

    plt.xscale("log")
    plt.xlabel("max_iter")
    plt.ylabel("Validation loss")

    plt.title("MSE: Loss vs max_iter")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_heatmap(max_iters, gammas, loss_te):
    """Display the complete hyperparameter grid"""
    img = plt.imshow(loss_te, aspect="auto", interpolation="nearest")

    plt.colorbar(img, label="loss")

    plt.xticks(
        np.arange(len(gammas)), ["%.1e" % gamma for gamma in gammas], rotation=45
    )
    plt.yticks(np.arange(len(max_iters)), max_iters)

    plt.xlabel("gamma")
    plt.ylabel("max iters")

    plt.title("MSE gradient descent: Hyperparameter Search")

    plt.tight_layout()
    plt.show()
