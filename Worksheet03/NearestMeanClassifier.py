import numpy as np
import matplotlib.pyplot as plt

import general as gen


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
        distance2mean[:, label] = gen.distance_from_mean(test_features, mean_points[label])

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
