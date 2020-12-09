# Data
from sklearn.datasets import load_digits
from sklearn import model_selection

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


def backward_search(data, target, significance_level=0.05):
    features = data.columns.tolist()
    while(len(features) > 2):
        features


def main():
    # Load data
    digits = load_digits()
    print(digits.keys())

    data = digits['data']
    images = digits['images']
    target = digits['target']
    target_names = digits['target_names']

    # Filter data
    filt_target_idx = np.argwhere((target == 1) | (target == 7))
    filt_target_names = target_names[(target_names == 1) | (target_names == 7)]
    filt_data = np.squeeze(data[filt_target_idx])
    filt_images = np.squeeze(images[filt_target_idx])
    filt_target = np.squeeze(target[filt_target_idx])

    # Split data (N_train/N_test = 3/2)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        filt_data,
        filt_target,
        test_size=0.35,
        random_state=0
    )

    matrix = X_train[20].reshape((8, 8))
    print(matrix)
    plt.imshow(matrix)
    plt.show()


if __name__ == '__main__':
    main()
