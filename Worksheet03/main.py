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


def reduce_dim(x):
    """
    Perform a user defined dimension reduction
    :param x: ndarray of size (N, 64)
    :return: ndarray of size (N, 2)
    """
    pass


def worse_reduce_dim(x):
    """
    Perform a user defined dimension reduction
    :param x: ndarray of size (N, 64)
    :return: ndarray of size (N, 2)
    """
    pass


def main():
    # Load data
    digits = load_digits()

    # Filtering data
    x_training, x_test, y_training, y_test = gen.data_preparation(digits, 0.33, 0)

    # Dimension reduction
    xr_training, xr_test = reduce_dim(x_training), reduce_dim(x_test)


if __name__ == '__main__':
    main()
