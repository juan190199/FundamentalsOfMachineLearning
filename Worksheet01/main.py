import matplotlib.pyplot as plt
import numpy as np


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
            return 1 / 2
        else:
            return np.random.randint(0, 2, len(x))
    if type == 'D':
        if error is True:
            return 1 / 2
        else:
            return np.ones(len(x))


def thresholding_error(N, threshold=None, plot=False, dataset=None):
    if threshold is not None:
        classifiers = np.array(['A', 'B'])
        thr_dict = {}
        for i in range(len(threshold)):
            thr_dict['Threshold: ' + str(threshold[i])] = np.zeros(shape=(4, len(N)))
        error = np.zeros(shape=(len(threshold), 4, len(N)))
        for j in range(len(threshold)):
            for i in range(10):
                for k in range(len(N)):
                    data = np.zeros(shape=(N[k], 2))
                    data = create_data(N[k])
                    mean = np.mean(data)
                    std = np.std(data)

                    # Analytical error
                    a_error_A = threshold_classifier(data[:, 0], type='A', threshold=threshold[j], error=True)
                    a_error_B = threshold_classifier(data[:, 0], type='B', threshold=threshold[j], error=True)

                    # Numerical error
                    prediction_A = threshold_classifier(data[:, 0], type='A', threshold=threshold[j])
                    prediction_B = threshold_classifier(data[:, 0], type='B', threshold=threshold[j])

                    n_error_A = np.sum(np.abs(np.subtract(prediction_A, data[:, 1])))
                    n_error_B = np.sum(np.abs(np.subtract(prediction_B, data[:, 1])))

                    num_error_A = n_error_A / (N[k])
                    num_error_B = n_error_B / (N[k])

                    print("Threshold: {}, dataset: {}, batch_size: {}".format(threshold[j], i, N[k]))
                    print("Analytical error classifier A: {}".format(a_error_A))
                    print("Numerical error classifier A: {}".format(num_error_A))
                    print("Analytical error classifier B: {}".format(a_error_B))
                    print("Numerical error classifier B: {} \n".format(num_error_B))

                    if plot is True:
                        if dataset == i:
                            thr_dict['Threshold: ' + str(threshold[j])][0, k] = a_error_A
                            thr_dict['Threshold: ' + str(threshold[j])][1, k] = num_error_A
                            thr_dict['Threshold: ' + str(threshold[j])][2, k] = a_error_B
                            thr_dict['Threshold: ' + str(threshold[j])][3, k] = num_error_B

        plot_thresholding_error(thr_dict, N, classifiers, threshold)
    else:
        classifiers = np.array(['C', 'D'])
        error = np.zeros(shape=(4, len(N)))
        for i in range(10):
            for k in range(len(N)):
                data = np.zeros(shape=(N[k], 2))
                data = create_data(N[k])
                mean = np.mean(data)
                std = np.std(data)
                # Analytical error
                a_error_C = threshold_classifier(data[:, 0], type='C', error=True)
                a_error_D = threshold_classifier(data[:, 0], type='D', error=True)
                # Numerical error
                prediction_C = threshold_classifier(data[:, 0], type='C')
                prediction_D = threshold_classifier(data[:, 0], type='D')

                n_error_C = np.sum(np.abs(np.subtract(prediction_C, data[:, 1])))
                n_error_D = np.sum(np.abs(np.subtract(prediction_D, data[:, 1])))

                num_error_C = n_error_C / (N[k])
                num_error_D = n_error_D / (N[k])

                print("Dataset: {}, batch_size: {}".format(i, N[k]))
                print("Analytical error classifier C: {}".format(a_error_C))
                print("Numerical error classifier C: {}".format(num_error_C))
                print("Analytical error classifier D: {}".format(a_error_D))
                print("Numerical error classifier D: {} \n".format(num_error_D))

                if plot is True:
                    if dataset == i:
                        error[0, k] = a_error_C
                        error[1, k] = num_error_C
                        error[2, k] = a_error_D
                        error[3, k] = num_error_D


def plot_thresholding_error(error, batch_size, classifiers, threshold=None, std=None):
    fig, ax = plt.subplots(1, len(threshold), figsize=(12, 3))
    errors = ['Analytical error classifier A', 'Numerical error classifier A', 'Analytical error classifier B',
              'Numerical error classifier B']
    if threshold is not None:
        for i in range(len(threshold)):
            # Dictionary to array
            a_error_A = error['Threshold: ' + str(threshold[i])][0, :]
            num_error_A = error['Threshold: ' + str(threshold[i])][1, :]
            a_error_B = error['Threshold: ' + str(threshold[i])][2, :]
            num_error_B = error['Threshold: ' + str(threshold[i])][3, :]

            x = np.linspace(0, batch_size[-1], 4)
            ax[i].plot(x, a_error_A, x, num_error_A, x, a_error_B, x, num_error_B)
            ax[i].set_title("Threshold {}".format(threshold[i]), fontsize=8.5)
            ax[i].legend(errors, fontsize='xx-small')
            ax[i].tick_params(axis='both', labelsize=5)
            ax[i].set_xlabel("Batch size", fontsize=7)
            ax[i].set_xlim(0, batch_size[-1])
            ax[i].set_xticks(batch_size)
            ax[i].set_ylabel("Error", fontsize=7)
            ax[i].set_ylim(0, 1)
            ax[i].set_yticks(np.linspace(0, 1, 5))

    # plt.title("Error wrt. batch_size given classifier type", loc='center', pad=3)
    plt.subplots_adjust(wspace=0.5)
    plt.show()


def main():
    # Task 1
    # data = create_data(500)
    # plot_data(data)

    # Task 2
    threshold = np.array([0.2, 0.5, 0.6])
    batch_size = np.array([10, 100, 1000, 10000])
    thresholding_error(batch_size, threshold, plot=True, dataset=1)

    # Task 3
    thresholding_error(batch_size)

    # Task 4


if __name__ == '__main__':
    main()
