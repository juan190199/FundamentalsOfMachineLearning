import numpy as np
from matplotlib import pyplot as plt


def create_data(N):
    Y = np.random.randint(0, 2, size=N)  # Sample instance labels from prior 1/2
    if N == 2:
        while np.all(Y == Y[0]):
            Y = np.random.randint(0, 2, size=N)  # Sample instance labels from prior 1/2

    u = np.random.uniform(size=N)
    X = np.zeros(N)

    for i in range(N):
        if Y[i] == 0:
            X[i] = 1 - np.sqrt(1 - u[i])
        else:
            X[i] = np.array(np.sqrt(u[i]))

    data_set = np.stack((X, Y), axis=1)
    return data_set


def plot_data(data):
    fig, ax = plt.subplots(1, 2, figsize=(20,20))
    ax[0].scatter(data[:, 1], data[:, 0], alpha=0.3, color='black')
    ax[0].set_title("Scatter of the data")
    ax[0].set_xlabel("Classes")
    ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylabel("Data points")
    ax[0].set_ylim(-0.5, 1.5)
    ax[1].hist2d(data[:, 1], data[:, 0], bins=(2, 20))
    ax[1].set_title("Histogram of data per class")
    plt.show()


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.subtract(x1, x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(np.subtract(x1, x2)))


def weighted_euclidean_distance():
    pass

