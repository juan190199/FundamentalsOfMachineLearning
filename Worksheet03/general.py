# Numbers
import numpy as np

# Plot
import matplotlib.pyplot as plt

# Methods
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

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


def log_likelihood(x, mu, cov, p):
    """
    Computes log-likelihood for each data point x
    :param x: ndarray: (N, 2)
    :param mu: ndarray: (2,)
    :param cov: ndarray: (2, 2)
    :param p: float
    :return: logl: ndarray: (N, 2)
    """
    alpha = - 0.5 * np.log(2 * np.pi * np.linalg.det(cov)) + np.log(p)
    x = x - mu
    logl = -0.5 * (np.sum(x.T * np.dot(np.linalg.inv(cov), x.T), axis=0)) + alpha
    return logl


def plot_ellipse_axis(mu, cov, color="blue"):
    """
    Plots main axis of the distribution given mean and covariance matrix
    :param mu:
    :param cov:
    :param color:
    :return:
    """
    # Eigenvalues/Eigenvector decomposition
    [lamb1, lamb2], [vec_1, vec_2] = np.linalg.eig(cov)
    lamb1, lamb2 = np.sqrt(lamb1), np.sqrt(lamb2)

    # Plot axis 1
    x1, y1 = ([mu[0] - lamb1 * vec_1[0], mu[0] + lamb1 * vec_1[0]],
              [mu[1] - lamb1 * vec_2[0], mu[1] + lamb1 * vec_2[0]])
    plt.plot(x1, y1, color)

    # Plot axis 2
    x2, y2 = ([mu[0] - lamb2 * vec_1[1], mu[0] + lamb2 * vec_1[1]],
              [mu[1] - lamb2 * vec_2[1], mu[1] + lamb2 * vec_2[1]])
    plt.plot(x2, y2, color)


def cross_validation(digits, fit_func, pred_func, num_sample=10):
    """
    Measure the correct accuracy with cross validation
    """
    # Get data
    data = digits["data"]
    target = digits["target"]

    # Need to prepare data again since 'data_preparation' returns test - train split
    # Data filering
    num_1, num_2 = 1, 7
    mask = np.logical_or(target == num_1, target == num_2)
    data = data[mask] / data.max()
    target = target[mask]

    # Relabel targets
    target[target == num_1] = 0
    target[target == num_2] = 1

    # Splits
    k_folds = KFold(n_splits=num_sample)

    mean_rate = np.zeros(num_sample)
    for i, (train, test) in enumerate(k_folds.split(data)):
        xr_train, xr_test = reduce_dim(data[train]), reduce_dim(data[test])
        mu, cov, p = fit_func(xr_train, target[train])
        predicted_labels = pred_func(mu, cov, p, xr_test)
        mean_rate[i] = np.mean(predicted_labels == target[test])

    print("Mean Accuracy Cross Validation: %f +/- %f" % (np.mean(mean_rate), np.std(mean_rate)))
