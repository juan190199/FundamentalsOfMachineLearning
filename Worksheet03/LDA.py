import numpy as np
import matplotlib.pyplot as plt

import general as gen


def fit_lda(training_features, training_labels):
    """
    Computes for each class: mean, and priors; and global covariance matrix
    :param training_features: ndarray: (N_training, 2)
    :param training_labels: ndarray: (N_training, 1)
    :return: mu: ndarray: (N_labels, 2), cov: ndarray: (2, 2), p: ndarray: (N_labels, 1)
    """
    mu, cov, p = [], [], []
    for label in np.unique(training_labels):
        # filtering the correct class
        data = training_features[training_labels == label]

        # mean
        mean = np.mean(data, axis=0)
        mu.append(mean)

        # Priors
        p.append(data.shape[0] / training_features.shape[0])

    # Global Variance, subtract class means first

    # computed as in texbook
    # global_feat = training_features - np.array(mu)[training_labels]
    # np.mean(training_features, axis=0)
    # cov = np.dot(global_feat.T, global_feat)/training_features.shape[0]

    # as numpy oneliner
    cov = np.cov(training_features.T)
    return mu, cov, p


def predict_lda(mu, cov, p, test_features):
    """
    This function returns the LDA predictions given
    as input lists of means, priors and the global covariance matrix
    input test_features: N_test x 2 numpy array
    output: test_predictions: N_test numpy array
    """
    loglikelihood = np.zeros((test_features.shape[0], len(mu)))

    # Find the loglikelihood for each test point
    for label in range(len(mu)):
        loglikelihood[:, label] = gen.log_likelihood(test_features,
                                                 mu[label],
                                                 cov,
                                                 p[label])
    return np.argmax(loglikelihood, axis=-1)


def visualization_LDA(xr_train, y_train, xr_test, y_test, mu, cov, p, mean_points, simple=True):
    """
    This Function visualizes the decision regions for the LDA Classifier
    """
    # Build Grid
    feat_min, feat_max = np.min(xr_test, axis=0), np.max(xr_test, axis=0)
    x, y = np.linspace(feat_min[0], feat_max[0], 200), np.linspace(feat_max[1], feat_min[1], 200)
    xx = np.array(np.meshgrid(x, y)).reshape(2, -1).T

    # Predict grid labels
    predicted_labels = predict_lda(mu, cov, p, xx)

    plt.figure(figsize=(16, 9))
    plt.title("Decision Regions (Blu: 1, Red: 7)")
    plt.imshow(predicted_labels.reshape(-1, 200),
               cmap="prism_r", alpha=0.2, vmin=-5,
               extent=(feat_min[0], feat_max[0], feat_min[1], feat_max[1]))

    # Scatter data
    plt.scatter(xr_test[y_test == 0, 0], xr_test[y_test == 0, 1], marker="o", s=30, c="b", label="1")
    plt.scatter(xr_test[y_test == 1, 0], xr_test[y_test == 1, 1], marker="x", s=30, c="r", label="7")

    # Scatter Mean
    plt.scatter(mean_points[0][0], mean_points[0][1], marker="o", s=50, c="g", label="Mean")
    plt.scatter(mean_points[1][0], mean_points[1][1], marker="o", s=50, c="g")

    # if simple == False add isocontours and clusters axiss
    if not simple:
        zz = np.exp(gen.log_likelihood(xx, mu[0], cov, p[0]))
        plt.contour(x, y, zz.reshape(-1, 200), 3, colors='blue')

        zz = np.exp(gen.log_likelihood(xx, mu[1], cov, p[1]))
        plt.contour(x, y, zz.reshape(-1, 200), 3, colors='red')

        gen.plot_ellipse_axis(mu[0], cov, color="green")
        gen.plot_ellipse_axis(mu[1], cov, color="green")

    plt.xlim(feat_min[0], feat_max[0])
    plt.ylim(feat_min[1], feat_max[1])

    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.legend()
    plt.show()


def cross_validation_lda(digits, num_sample=10):
    """
    Measure the correct accuracy with cross validation
    """
    mean_rate = np.zeros(num_sample)
    for i in range(num_sample):
        x_train, x_test, y_train, y_test = gen.data_preparation(digits, test_percentage=0.33, random_seed=None)

        xr_train, xr_test = gen.reduce_dim(x_train), gen.reduce_dim(x_test)
        mu, cov, p = fit_lda(xr_train, y_train)
        predicted_labels = predict_lda(mu, cov, p, xr_test)
        mean_rate[i] = np.mean(predicted_labels == y_test)

    print("Mean Accuracy Cross Validation: %f +/- %f" % (np.mean(mean_rate), np.std(mean_rate)))

