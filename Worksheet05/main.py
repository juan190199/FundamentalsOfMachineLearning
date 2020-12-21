import numpy as np
from sklearn.datasets import load_digits

import Worksheet05.general as gen
import Worksheet05.DensityTree as DensityTree


def task1():
    pass


def main():
    # Load digits
    digits = load_digits()

    # Data preparation
    x_training, x_test, y_training, y_test = gen.data_preparation(digits, 0.33, 0)



if __name__ == '__main__':
    main()
