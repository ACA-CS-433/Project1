"""Code in this file has been written with the help of the ML labs code & LLM"""

import matplotlib.pyplot as plt
import numpy as np


def plot_vs_gamma(gammas, max_iters, value, metric):
    """Plots the given value as a function gamma"""
    plt.figure()
    for i, max_iter in enumerate(max_iters):
        plt.plot(gammas, value[i, :], marker="o", label=f"max_iters ={max_iter} ")

    plt.xscale("log")

    plt.xlabel("gamma")
    plt.ylabel(metric)

    plt.title(f"MSE: validation {metric} vs Gamma")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_vs_max_iters(max_iters, gammas, value, metric):
    """Plots the given value as a function max_iters"""
    plt.figure()
    for j, gamma in enumerate(gammas):
        plt.plot(max_iters, value[:, j], marker="o", label=f"gamma = {gamma:.1e}")

    plt.xscale("log")

    plt.xlabel("max_iter")
    plt.ylabel(metric)

    plt.title(f"MSE: validation {metric} vs max_iter")

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_heatmap(max_iters, gammas, loss_te, metric):
    """Display the complete hyperparameter grid"""
    plt.figure()
    img = plt.imshow(loss_te, aspect="auto", interpolation="nearest")

    plt.colorbar(img, label=metric)

    plt.xticks(
        np.arange(len(gammas)), ["%.1e" % gamma for gamma in gammas], rotation=45
    )
    plt.yticks(np.arange(len(max_iters)), max_iters)

    plt.xlabel("gamma")
    plt.ylabel("max iters")

    plt.title(f"MSE gradient descent: Hyperparameter Search for {metric}")

    plt.tight_layout()
    plt.show()
