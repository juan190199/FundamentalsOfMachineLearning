# Data
from sklearn.datasets import load_digits
from sklearn import model_selection

import Worksheet03.general as gen

# Numbers
import numpy as np
import numpy.testing as nt

# Stats
from sklearn.preprocessing import StandardScaler

# Data frame
import pandas as pd

# Plots
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'


def reduce_dim(x):
    """
    Perform a user defined dimension reduction
    :param x: ndarray of size (N, 64)
    :return: ndarray of size (N, 2)
    """
    # mean(Upper part) - mean(Lower part)
    print(x[:, :x.shape[-1] // 4].shape)
    feature1 = (np.mean(x[:, :x.shape[-1] // 4], axis=-1) - np.mean(x[:, 3 * x.shape[-1] // 4:], axis=-1))

    # mean(Upper part) * mean(Lower part)
    feature2 = (np.mean(x[:, :x.shape[-1] // 4] * x[:, 3 * x.shape[-1] // 4:], axis=-1))

    return np.array([feature1, feature2]).T


def scatter_plot_simple(x, y, title="Training"):
    """
    Dataset scatter plot
    :param x: ndarray: (N, 2)
    :param y: ndarray: (N, 1)
    :param title: Plot title
    :return: None
    """
    plt.figure(figsize=(16, 9))
    plt.title(title + " Scatter plot: 1 vs 7")

    plt.scatter(x[y == 0, 0], x[y == 0, 1], marker="o", s=30, c="b", label="1")
    plt.scatter(x[y == 1, 0], x[y == 1, 1], marker="x", s=30, c="r", label="7")

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.show()


def worse_reduce_dim(x):
    """
    Perform a user defined dimension reduction
    :param x: ndarray of size (N, 64)
    :return: ndarray of size (N, 2)
    """
    # mean(Image)
    feat1 = np.mean(x, axis=-1)

    # var(Image)
    feat2 = np.var(x, axis=-1)

    return np.array([feat1, feat2]).T


def main():
    # Load data
    digits = load_digits()

    # Filtering data
    x_training, x_test, y_training, y_test = gen.data_preparation(digits, 0.33, 0)

    # Dimension reduction
    xr_training, xr_test = reduce_dim(x_training), reduce_dim(x_test)

    # Scatter plot
    scatter_plot_simple(xr_training, y_training, "Training")
    scatter_plot_simple(xr_test, y_test, "Test")

    # Dimension Reduction
    _xr_training, _xr_test = worse_reduce_dim(x_training), worse_reduce_dim(x_test)

    # Scatter Plot
    scatter_plot_simple(_xr_training, y_training, "Training - Worse features")


if __name__ == '__main__':
    main()
