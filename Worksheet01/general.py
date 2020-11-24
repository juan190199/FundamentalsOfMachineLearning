import numpy as np
from matplotlib import pyplot as plt


def create_data(N):
    """
    Create data using inverse transform method (Explained in Task1.pdf)
    :param N: batch size of the data to be created
    :return:
    """
    Y = np.random.randint(0, 2, size=N)  # Sample instance labels from prior 1/2
    if N == 2:
        while np.all(Y == Y[0]):
            Y = np.random.randint(0, 2, size=N)  # Sample instance labels from prior 1/2

    u = np.random.uniform(size=N)
    X = np.zeros(N)
    idx0 = Y == 0
    idx1 = ~idx0
    X[idx0] = 1 - np.sqrt(1 - u[idx0])
    X[idx1] = np.sqrt(u[idx1])
    data_set = np.stack((X, Y), axis=1)
    return data_set


def plot_data(data):
    X_0 = data[data[:, 1] == 0][:, 0]
    X_1 = data[data[:, 1] == 1][:, 0]
    fig, ax = plt.subplots(1, 5, figsize=(15, 10))
    ax[0].scatter(data[:, 1], data[:, 0], alpha=0.3, color='black')
    ax[0].set_title("Scatter of the data")
    ax[0].set_xlabel("Classes")
    ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylabel("Data points")
    ax[0].set_ylim(-0.5, 1.5)

    ax[1].hist2d(data[:, 1], data[:, 0], bins=(2, 20))
    ax[1].set_title("Histogram of data per class")

    ax[2].bar([0, 1], [X_0.size, X_1.size], width=0.6)
    ax[2].set_xticks([0, 1]);
    ax[2].set_xlim([-0.5, 1.5])
    ax[2].set_yticks([0, 25000, 50000], ['0', '25k', '50k'])
    ax[2].set_title(r'Prior for class $Y$')

    ax[3].hist(X_0, 50, density=True, facecolor='green', alpha=0.5)
    ax[3].set_ylabel(r'$p(X = x \mid Y = 0)$')
    ax[3].plot([0, 1], [2, 0])
    ax[3].set_title(r'Likelihood for $Y=0$')

    ax[4].hist(X_1, 50, density=True, facecolor='blue', alpha=0.5)
    ax[4].set_ylabel(r'$p(X = x \mid Y = 1)$')
    ax[4].plot([0, 1], [0, 2])
    ax[4].set_title(r'Likelihood for $Y=1$')

    plt.show()


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(np.subtract(x1, x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sum(np.abs(np.subtract(x1, x2)))


def weighted_euclidean_distance():
    pass


if __name__ == '__main__':
    main()
