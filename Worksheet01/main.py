import matplotlib.pyplot as plt
import numpy as np


def create_data(N):
    Y = np.random.randint(0, 2, size=N)  # Sample instance labels from prior 1/2]
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
    fig, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 1], data[:, 0], alpha=0.3, color='black')
    ax[0].set_title("Scatter of the data")
    ax[0].set_xlabel("Classes")
    ax[0].set_xlim(-0.5, 1.5)
    ax[0].set_ylabel("Data points")
    ax[0].set_ylim(-0.5, 1.5)

    ax[1].hist2d(data[:, 1], data[:, 0], bins=(2, 20))
    ax[1].set_title("Histogram of data per class")

    plt.show()


def threshold_classifier(x, type, threshold=None, error=False):
    if type == 'A':
        if error is True:
            return 1 / 4 + (threshold - 1 / 2) ** 2
        else:
            binary_arr = np.where(x < threshold, 0, 1)
            return binary_arr
    if type == 'B':
        if error is True:
            return 3 / 4 - (threshold - 1 / 2) ** 2
        else:
            binary_arr = np.where(x < threshold, 1, 0)
            return binary_arr
    if type == 'C':
        if error is True:
            return 1/2
        else:
            return np.random.randint(0, 2, len(x))
    if type == 'D':
        if error is True:
            return 1/2
        else:
            return np.ones(len(x))


# ToDo: Plot errors and standard deviation
def thresholding_error(N, threshold=None):
    for j in range(len(threshold)):
        for i in range(10):
            for k in range(len(N)):
                data = np.zeros(shape=(N[k], 2))
                data = create_data(N[k])
                # Analytical error
                a_error_A = threshold_classifier(data[:, 0], type='A', threshold=threshold[j], error=True)
                a_error_B = threshold_classifier(data[:, 0], type='B', threshold=threshold[j], error=True)
                a_error_C = threshold_classifier(data[:, 0], type='C', threshold=threshold[j], error=True)
                a_error_D = threshold_classifier(data[:, 0], type='D', threshold=threshold[j], error=True)
                # Numerical error
                prediction_A = threshold_classifier(data[:, 0], type='A', threshold=threshold[j])
                prediction_B = threshold_classifier(data[:, 0], type='B', threshold=threshold[j])
                prediction_C = threshold_classifier(data[:, 0], type='C', threshold=threshold[j])
                prediction_D = threshold_classifier(data[:, 0], type='D', threshold=threshold[j])

                n_error_A = np.sum(np.abs(np.subtract(prediction_A, data[:, 1])))
                n_error_B = np.sum(np.abs(np.subtract(prediction_B, data[:, 1])))
                n_error_C = np.sum(np.abs(np.subtract(prediction_C, data[:, 1])))
                n_error_D = np.sum(np.abs(np.subtract(prediction_D, data[:, 1])))

                num_error_A = n_error_A / (N[k])
                num_error_B = n_error_B / (N[k])
                num_error_C = n_error_C / (N[k])
                num_error_D = n_error_D / (N[k])

                print("Threshold: {}, dataset: {}, batch_size: {}".format(threshold[j], i, N[k]))
                print("Analytical error classifier A: {}".format(a_error_A))
                print("Numerical error classifier A: {}".format(num_error_A))
                print("Analytical error classifier B: {}".format(a_error_B))
                print("Numerical error classifier B: {}".format(num_error_B))
                print("Analytical error classifier C: {}".format(a_error_C))
                print("Numerical error classifier C: {}".format(num_error_C))
                print("Analytical error classifier D: {}".format(a_error_D))
                print("Numerical error classifier D: {}".format(num_error_D))


def main():
    # Task 1
    data = create_data(500)
    plot_data(data)

    # Task 2 and 3
    threshold = np.array([0.2, 0.5, 0.6])
    N = np.array([10, 100, 1000, 10000])
    thresholding_error(N, threshold)


if __name__ == '__main__':
    main()
