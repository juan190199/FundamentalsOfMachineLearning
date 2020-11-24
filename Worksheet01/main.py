import general as gen

import numpy as np
import pandas as pd
from IPython.display import display


def threshold_classifier(X, type, threshold=None, error=False):
    """
    :param X: Numpy array with test set to be classified
    :param type: Type of classifier to be applied
    :param threshold: Threshold of the classifier
    :param error: Boolean variable to determine the error of the classifier given the type and the threshold
    :return: If error is false, return predictions for test set. Otherwise, return error for the given classifier type and threshold
    """
    if type == 'A':
        if error is True:
            return 1 / 4 + (threshold - 1 / 2) ** 2
        else:
            binary_arr = np.where(X < threshold, 0, 1)
            return binary_arr
    if type == 'B':
        if error is True:
            return 3 / 4 - (threshold - 1 / 2) ** 2
        else:
            binary_arr = np.where(X < threshold, 1, 0)
            return binary_arr
    if type == 'C':
        if error is True:
            return 1 / 2
        else:
            return np.random.randint(0, 2, len(X))
    if type == 'D':
        if error is True:
            return 1 / 2
        else:
            return np.ones(len(X))


def calculate_ose(type, batch, n_data_sets, threshold=None):
    """
    Calculate out-of-sample errors given number of data sets
    :param type: Classifier type
    :param batch: size of the test set
    :param n_data_sets:
    :param threshold: threshold of the classifier
    :return:
    """
    oses = np.zeros(n_data_sets)
    for i in range(n_data_sets):
        test_set = gen.create_data(batch)
        prediction = threshold_classifier(test_set[:, 0], type=type, threshold=threshold)
        n_errors = np.sum(np.abs(np.subtract(prediction, test_set[:, 1])))
        ose = n_errors / batch
        oses[i] = ose

    return oses


def task1():
    data = gen.create_data(30000)
    gen.plot_data(data)


def task2():
    thresholds = [0.2, 0.5, 0.6]
    batch_sizes = [10, 100, 10000, 10000]

    classifier_A = []
    classifier_B = []

    for threshold in thresholds:
        for batch in batch_sizes:
            error_rates_A = calculate_ose(type='A', batch=batch, n_data_sets=10, threshold=threshold)
            classifier_A.append({"$x_0$": threshold,
                                 "Batch size": batch,
                                 "Mean": error_rates_A.mean(),
                                 "Std": error_rates_A.std()
                                 })

            error_rates_B = calculate_ose(type='B', batch=batch, n_data_sets=10, threshold=threshold)
            classifier_B.append({"$x_0$": threshold,
                                 "Batch size": batch,
                                 "Mean": error_rates_B.mean(),
                                 "Std": error_rates_B.std()
                                 })

    df_A = pd.DataFrame(classifier_A)
    df_A = df_A.groupby(["$x_0$", "Batch size"]).first().unstack()
    df_A


def task3():
    pass


def task4():
    pass


def main():
    # task1()
    task2()
    # task3()
    # task4()


if __name__ == '__main__':
    main()
