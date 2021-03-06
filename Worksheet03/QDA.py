import numpy as np
import matplotlib.pyplot as plt

import Worksheet03.general as gen

plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.titleweight'] = 'bold'


def fit_qda(training_features, training_labels):
    """
    Computes for each class: mean, covariance matrix and priors
    :param training_features: ndarray: (N_training, num_features)
    :param training_labels: ndarray: (N_training, 1)
    :return: mu: ndarray: (N_labels, num_features), cov: ndarray: (N_labels, num_features, num_features),
    p: ndarray: (N_labels, 1)
    """
    mu, cov, p = [], [], []
    for label in np.unique(training_labels):
        # filtering the correct class
        data = training_features[training_labels == label]

        # mean
        mean = np.mean(data, axis=0)
        mu.append(mean)

        # Covariance
        # Computed as in textbook
        # data_centered = data - mean
        # cov.append(np.dot(data_centered.T, data_centered)/data.shape[0])

        # as numpy oneliner
        cov.append(np.cov(data.T))

        # Prior
        p.append(data.shape[0] / training_features.shape[0])

    return mu, cov, p


def predict_qda(mu, cov, p, test_features):
    """
    Computes QDA prediction given test_features
    :param mu: ndarray: (2,)
    :param cov: ndarray: (2, 2)
    :param p: float
    :param test_features: ndarray: (N_test, 2)
    :return: test_predictions: ndarray(N_test, )
    """
    loglikelihood = np.zeros((test_features.shape[0], len(mu)))

    # Find the loglikelihood for each test point
    for label in range(len(mu)):
        loglikelihood[:, label] = gen.log_likelihood(test_features,
                                                 mu[label],
                                                 cov[label],
                                                 p[label])
    return np.argmax(loglikelihood, axis=-1)


def visualization_QDA(xr_train, y_train, xr_test, y_test, mu, cov, p, mean_points, simple=True):
    """
    This Function visualizes the decision regions for the QDA Classifier
    """
    # Build Grid
    feat_min, feat_max = np.min(xr_test, axis=0), np.max(xr_test, axis=0)

    x, y = np.linspace(feat_min[0], feat_max[0], 200), np.linspace(feat_max[1], feat_min[1], 200)
    xx = np.array(np.meshgrid(x, y)).reshape(2, -1).T

    # Predict grid labels
    predicted_labels = predict_qda(mu, cov, p, xx)

    # Decison boundary
    plt.figure(figsize=(16, 9))
    plt.title("Decision Regions (Blue: 1, Red: 7)")
    plt.imshow(predicted_labels.reshape(-1, 200),
               cmap="prism_r",
               alpha=0.2, vmin=-5,
               extent=(feat_min[0], feat_max[0], feat_min[1], feat_max[1]))

    # Scatter data
    plt.scatter(xr_test[y_test == 0, 0], xr_test[y_test == 0, 1], marker="o", s=30, c="b", label="1")
    plt.scatter(xr_test[y_test == 1, 0], xr_test[y_test == 1, 1], marker="x", s=30, c="r", label="7")

    # Scatter mean points
    plt.scatter(mean_points[0][0], mean_points[0][1], marker="o", s=50, c="g", label="Mean")
    plt.scatter(mean_points[1][0], mean_points[1][1], marker="o", s=50, c="g")

    # if simple == False add isocontours and clusters axiss
    if not simple:
        zz = np.exp(gen.log_likelihood(xx, mu[0], cov[0], p[0]))
        plt.contour(x, y, zz.reshape(-1, 200), 3, colors='blue')

        zz = np.exp(gen.log_likelihood(xx, mu[1], cov[1], p[1]))
        plt.contour(x, y, zz.reshape(-1, 200), 3, colors='red')

        gen.plot_ellipse_axis(mu[0], cov[0], color="green")
        gen.plot_ellipse_axis(mu[1], cov[1], color="green")

    plt.xlim(feat_min[0], feat_max[0])
    plt.ylim(feat_min[1], feat_max[1])

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.show()

