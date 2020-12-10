# Data
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

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


def data_preparation(digits, test_percentage=0.33, random_seed=0):
    """
    Filter the digits (1, 7) from the data set and randomly splits it in train and test set.
    :param digits:
    :param test_percentage:
    :param random_seed:
    :return:
    """
    data = digits['data']
    target = digits['target']

    # Data filtering
    num_1, num_2 = 1, 7
    mask = np.logical_or(target == num_1, target == num_2)
    data = data[mask] / data.max()
    target = target[mask]

    # Relabel targets
    target[target == num_1] = 0
    target[target == num_2] = 1

    # Random split
    X_train, X_test, Y_train, Y_test = train_test_split(
        data,
        target,
        test_size=test_percentage,
        random_state=random_seed
    )

    return X_train, X_test, Y_train, Y_test


def reduce_dim(x):
    """
    Perform a user defined dimension reduction
    :param x: ndarray of size (N, 64)
    :return: ndarray of size (N, 2)
    """
    # mean(Upper part) - mean(Lower part)
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
    # plt.show()


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


def distance_from_mean(x, mean):
    """
    Computes L2 distance to the mean
    :param x:
    :param mean:
    :return:
    """
    return np.sqrt(np.sum((x - mean) ** 2, axis=-1))


def nearest_mean(training_features, training_labels, test_features):
    """
    This function returns the nearest mean predictions given input training_features, training_labels, and test_features
    :param training_features: ndarray: (N_training, 2)
    :param training_labels: ndarray: (N_training, 1)
    :param test_features: ndarray: (N_test, 2)
    :return: test_predictions: ndarray: (N_test, 1)
    """

    classes_list = np.unique(training_labels)
    mean_points = []
    # Find all mean points
    for label in classes_list:
        mean_points.append(np.mean(training_features[training_labels == label], axis=0))

    distance2mean = np.zeros((test_features.shape[0],
                              classes_list.shape[0]))
    # Compute the distances between test_features and all mean
    for label in classes_list:
        distance2mean[:, label] = distance_from_mean(test_features, mean_points[label])

    return np.argmin(distance2mean, axis=-1), mean_points


def visualization_NearestMean(xr_train, y_train, xr_test, y_test):
    """
    Visualization of decision regions for the Nearest Mean Classifier
    :param xr_train:
    :param y_train:
    :param xr_test:
    :param y_test:
    :return:
    """
    # Build Grid
    feat_min, feat_max = np.min(xr_test, axis=0), np.max(xr_test, axis=0)
    x, y = np.linspace(feat_min[0], feat_max[0], 2000), np.linspace(feat_max[1], feat_min[1], 2000)
    xx = np.array(np.meshgrid(x, y)).reshape(2, -1).T

    # Predict grid labels
    predicted_labels, mean_points = nearest_mean(xr_train, y_train, xx)

    # Decison boundary
    plt.figure(figsize=(16, 9))
    plt.title("Decision Regions (Blue: 1, Red: 7)")
    plt.imshow(predicted_labels.reshape(-1, 2000),
               cmap="prism_r", alpha=0.2, vmin=-5,
               extent=(feat_min[0], feat_max[0], feat_min[1], feat_max[1]))

    # Scatter data
    plt.scatter(xr_test[y_test == 0, 0], xr_test[y_test == 0, 1], marker="o", s=30, c="b", label="1")
    plt.scatter(xr_test[y_test == 1, 0], xr_test[y_test == 1, 1], marker="x", s=30, c="r", label="7")

    # Scatter mean points
    plt.scatter(mean_points[0][0], mean_points[0][1], marker="o", s=50, c="g", label="Mean")
    plt.scatter(mean_points[1][0], mean_points[1][1], marker="o", s=50, c="g")

    plt.xlim(feat_min[0], feat_max[0])
    plt.ylim(feat_min[1], feat_max[1])

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    # plt.show()


def fit_qda(training_features, training_labels):
    """
    Computes for each class: mean, covariance matrix and priors
    :param training_features: ndarray: (N_training, 2)
    :param training_labels: ndarray: (N_training, 1)
    :return: mu: ndarray: (N_labels, 2), cov: ndarray: (N_labels, 2, 2), p: ndarray: (N_labels, 1)
    """
    mu, cov, p = [], [], []
    for label in np.unique(training_labels):
        # Filtering the correct class
        data = training_features[training_labels == label]

        # Mean
        mean = np.mean(data, axis=0)
        mu.append(mean)

        # Covariance
        # Computed as in textbook
        # data_centered = data - mean
        # cov.append(np.dot(data_centered.T, data_centered)/data.shape[0])

        # As numpy oneliner
        cov.append(np.cov(data.T))

        # Prior
        p.append(data.shape[0]/training_features.shape[0])

        return mu, cov, p


def log_likelihood(x, mu, cov, p):
    pass


def predict_qda(mu, cov, p, test_features):
    pass


def main():
    # Load data
    digits = load_digits()

    # Filtering data
    x_training, x_test, y_training, y_test = data_preparation(digits, 0.33, 0)

    # Dimension reduction
    xr_training, xr_test = reduce_dim(x_training), reduce_dim(x_test)

    # Scatter plot
    scatter_plot_simple(xr_training, y_training, "Training")
    scatter_plot_simple(xr_test, y_test, "Test")

    # Dimension reduction
    _xr_training, _xr_test = worse_reduce_dim(x_training), worse_reduce_dim(x_test)

    # Scatter Plot
    scatter_plot_simple(_xr_training, y_training, "Training - Worse features")

    # Dimension reduction
    xr_test = reduce_dim(x_test)

    # Find nearest mean predictions
    predicted_labels, mean_points = nearest_mean(xr_training, y_training, xr_test)

    # Print accuracy
    print("Accuracy Nearest Mean: ", np.mean(predicted_labels == y_test))

    visualization_NearestMean(xr_training, y_training, xr_test, y_test)


if __name__ == '__main__':
    main()
