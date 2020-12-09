import Worksheet01.general as gen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def threshold_classifier(type, X=None, threshold=None, error=False):
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


def calculate_error_thresholding_classifier(type, error=False, batch=None, n_data_sets=None, threshold=None):
    """
    Calculate out-of-sample errors given number of data sets
    :param type: Classifier type
    :param error: Boolean variable. True to calculate analytical error of the classifier
    :param batch: size of the test set
    :param n_data_sets:
    :param threshold: threshold of the classifier
    :return:
    """
    if error is False:
        oses = np.empty(n_data_sets)
        for i in range(n_data_sets):
            test_set = gen.create_data(batch)
            prediction = threshold_classifier(type=type, X=test_set[:, 0], threshold=threshold)
            n_errors = np.sum(np.abs(np.subtract(prediction, test_set[:, 1])))
            ose = n_errors / batch
            oses[i] = ose
        return oses
    else:
        a_error = threshold_classifier(type=type, error=error, threshold=threshold)
        return a_error


def plot_error_rate(dfs, thresholds=None):
    if thresholds is None:
        df_C = dfs[0]
        df_D = dfs[1]
        df_C.plot(y="Mean", yerr="Std", logx=True, color="C0", ax=plt.gca(), label="Classifier C")
        df_D.plot(y="Mean", yerr="Std", logx=True, color="C1", ax=plt.gca(), label="Classifier D")
        plt.title("Error for classifier C and D")
        plt.show()
    else:
        cont_thresholds = np.linspace(0, 1, 100)
        error_A = calculate_error_thresholding_classifier(type='A', error=True, threshold=cont_thresholds)
        error_B = calculate_error_thresholding_classifier(type='B', error=True, threshold=cont_thresholds)
        errors = [error_A, error_B]

        for name, df, true_error in zip(["A", "B"], dfs, errors):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            plt.suptitle(f"Classifier {name}")

            plt.sca(ax1)
            plt.ylim(0, 1)
            plt.title("Error rate mean")
            df["Mean"].plot(marker="o", yerr=df["Std"], lw=1, ax=ax1)
            plt.plot(cont_thresholds, true_error, "k--")
            plt.xticks([0] + thresholds + [1])

            plt.sca(ax2)
            plt.title("Error rate std")
            df["Std"].T.plot(marker="o", ax=ax2, logx=True, logy=True)

            plt.tight_layout()

            plt.show()


def nn_classifier(training_set, test_set):
    """

    :return:
    """
    prediction = np.empty(test_set.shape[0])
    for i in range(test_set.shape[0]):
        diff = np.abs(training_set[:, 0] - test_set[i, 0])
        idx = np.argmin(diff)
        prediction[i] = training_set[idx, 1]
    return prediction


def calculate_error_nn_classifier(size_data, batch_size, n_data_sets):
    test_set = gen.create_data(batch_size)
    oses = np.empty(n_data_sets)
    for i in range(n_data_sets):
        data = gen.create_data(size_data)
        prediction = nn_classifier(data, test_set)
        n_errors = np.sum(np.abs(np.subtract(prediction, test_set[:, 1])))
        ose = n_errors / batch_size
        oses[i] = ose
    return oses


def task1():
    data = gen.create_data(30000)
    gen.plot_data(data)


def task2():
    thresholds = [0.2, 0.5, 0.6]
    batch_sizes = [10, 100, 1000, 10000]

    classifier_A = []
    classifier_B = []

    for threshold in thresholds:
        for batch in batch_sizes:
            error_rates_A = calculate_error_thresholding_classifier(type='A', batch=batch, n_data_sets=10,
                                                                    threshold=threshold)
            classifier_A.append({"$x_0$": threshold,
                                 "Batch size": batch,
                                 "Mean": error_rates_A.mean(),
                                 "Std": error_rates_A.std()
                                 })

            error_rates_B = calculate_error_thresholding_classifier(type='B', batch=batch, n_data_sets=10,
                                                                    threshold=threshold)
            classifier_B.append({"$x_0$": threshold,
                                 "Batch size": batch,
                                 "Mean": error_rates_B.mean(),
                                 "Std": error_rates_B.std()
                                 })

    df_A = pd.DataFrame(classifier_A)
    df_A = df_A.groupby(["$x_0$", "Batch size"]).first().unstack()
    print(df_A)

    df_B = pd.DataFrame(classifier_B)
    df_B = df_B.groupby(["$x_0$", "Batch size"]).first().unstack()
    print(df_B)

    plot_error_rate([df_A, df_B], thresholds)


def task3():
    batch_sizes = [10, 100, 1000, 10000]

    classifier_C = []
    classifier_D = []

    for batch in batch_sizes:
        error_rates_C = calculate_error_thresholding_classifier(type='C', batch=batch, n_data_sets=10)
        classifier_C.append({"Batch size": batch,
                             "Mean": error_rates_C.mean(),
                             "Std": error_rates_C.std()
                             })

        error_rates_D = calculate_error_thresholding_classifier(type='D', batch=batch, n_data_sets=10)
        classifier_D.append({"Batch size": batch,
                             "Mean": error_rates_D.mean(),
                             "Std": error_rates_D.std()
                             })

    df_C = pd.DataFrame(classifier_C)
    df_C = df_C.groupby(["Batch size"]).first()
    print(df_C)

    df_D = pd.DataFrame(classifier_D)
    df_D = df_D.groupby(["Batch size"]).first()
    print(df_D)

    plot_error_rate([df_C, df_D])


def task4():
    size_data = 2
    batch_size = 10000
    oses = calculate_error_nn_classifier(size_data=size_data, batch_size=batch_size, n_data_sets=300)
    print(f'Average error: {np.mean(oses):.3f}% ± {np.std(oses):.3f}%')

    size_data = 100
    oses = calculate_error_nn_classifier(size_data=size_data, batch_size=batch_size, n_data_sets=300)
    print(f'Average error: {np.mean(oses):.3f}% ± {np.std(oses):.3f}%')


def main():
    task1()
    task2()
    task3()
    task4()


if __name__ == '__main__':
    main()
