import general as gen

import numpy as np


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


def calculate_ose(test_set, prediction):
    """
    Calculate out-of-sample error
    :param test_set:
    :param prediction:
    :return:
    """
    n_errors = np.sum(np.abs(np.subtract(prediction, test_set[:, 1])))
    ose = n_errors/test_set.shape[0]
    return ose


def task1():
    data = gen.create_data(30000)
    gen.plot_data(data)


def task2():
    thresholds = [0.2, 0.5, 0.6]
    test_set = gen.create_data(1000)
    prediction = threshold_classifier(test_set[:, 0], type='A', threshold=0.5)
    ose = calculate_ose(test_set, prediction)
    print(ose)


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
