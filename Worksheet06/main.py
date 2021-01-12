import Worksheet06.Data.DataPreparation as DP
import Worksheet06.LinearRegression as LR
import Worksheet06.RegressionForest as RF

import numpy as np
import pandas as pd

from IPython.display import display
pd.options.display.float_format = '{:,.2f}'.format


def compute_error(model, test_features, test_labels):
    mean_squared_error = 0
    n = len(test_features)

    for i in range(n):
        mean_squared_error = mean_squared_error + (test_labels[i] - model.predict(test_features[i])) ** 2

    return mean_squared_error / n


def shuffle_data(features, feature_index):
    """
    Shuffles the data in the column denoted by feature_index. All other data remain unchanged
    :param features: ndarray: (n_instances, n_features)
    :param feature_index: entries in feature_index-th column will be shuffled randomly
    :return: data with shuffled feature_index
    """
    features = features.transpose()
    shuffled_feature = np.random.permutation(features[feature_index])
    features[feature_index] = shuffled_feature

    return features.transpose()


def task1(df, df_mean):
    Y = df["percentageReds"].to_numpy()
    X = df.drop(labels=["percentageReds"], axis=1).to_numpy()

    # Number of folds
    L = 10

    # Create  L folds
    N = len(X)
    indices = np.random.choice(N, N, replace=False)
    X_folds = np.array(np.array_split(X[indices], L), dtype=object)
    Y_folds = np.array(np.array_split(Y[indices], L), dtype=object)

    # Error Linear Regression
    error = []
    for i in range(L):
        print(i / L * 100, "%")
        # Create training and test data
        X_train = np.concatenate(X_folds[np.arange(L) != i], axis=0)
        Y_train = np.concatenate(Y_folds[np.arange(L) != i], axis=0)
        X_test = X_folds[i]
        Y_test = Y_folds[i]

        # Compute error
        regression = LR.LinearRegression(df_mean)
        regression.train(X_train, Y_train)
        error.append(compute_error(regression, X_test, Y_test))
    error = np.mean(error)

    # Print error
    print("\nError rate, linear regression:")
    print(error)

    # Error Regression Forest
    error = []
    for i in range(L):
        print(i / L * 100, "%")
        # Create training and test data
        X_train = np.concatenate(X_folds[np.arange(L) != i], axis=0)
        Y_train = np.concatenate(Y_folds[np.arange(L) != i], axis=0)
        X_test = X_folds[i]
        Y_test = Y_folds[i]

        # Compute error
        forest = RF.RegressionForest(5, df_mean)
        forest.train(X_train, Y_train, n_min=500)
        error.append(compute_error(forest, X_test, Y_test))
    error = np.mean(error)

    # Print error
    print("\nerror rate, regression forest:")
    print(error)


def task2(df, df_mean):
    color_rating_index = 8  # index of the color rating in df
    L = 10  # number of folds

    # Load csv-file where we save the mean squared errors
    err_data = pd.read_csv("errors.txt", sep=",", index_col=False)

    # Load original data set
    Y = df["percentageReds"].to_numpy()
    X = df.drop(labels=["percentageReds"], axis=1).to_numpy()

    # Linear Regression
    # Shuffle data
    Y_shuffled = Y
    X_shuffled = shuffle_data(X, 8)

    # Create L folds
    N = len(X_shuffled)
    indices = np.random.choice(N, N, replace=False)
    X_folds = np.array(np.array_split(X_shuffled[indices], L), dtype=object)
    Y_folds = np.array(np.array_split(Y_shuffled[indices], L), dtype=object)

    error = []
    for i in range(L):
        print(i / L * 100, "%")
        # create training and test data
        X_train = np.concatenate(X_folds[np.arange(L) != i], axis=0)
        Y_train = np.concatenate(Y_folds[np.arange(L) != i], axis=0)
        X_test = X_folds[i]
        Y_test = Y_folds[i]

        # Compute error
        regression = LR.LinearRegression(df_mean)
        regression.train(X_train, Y_train)
        error.append(compute_error(regression, X_test, Y_test))
    error_lr = np.mean(error)

    # Print error and save the value
    print("\nerror rate, linear regression:")
    print(error_lr)

    # Regression Tree
    error = []
    for i in range(L):
        print(i / L * 100, "%")
        # Create training and test data
        X_train = np.concatenate(X_folds[np.arange(L) != i], axis=0)
        Y_train = np.concatenate(Y_folds[np.arange(L) != i], axis=0)
        X_test = X_folds[i]
        Y_test = Y_folds[i]

        # Compute error
        forest = RF.RegressionForest(5, df_mean)
        forest.train(X_train, Y_train, n_min=500)
        error.append(compute_error(forest, X_test, Y_test))
    error = np.mean(error)

    # Print error and save the value
    print("\nerror rate, regression tree:")
    print(error)
    err_data.loc[len(err_data)] = [error_lr, error]

    err_data.to_csv("errors.txt", sep=",", index=False)

    err_data = pd.read_csv("errors.txt", sep=",", index_col=False)
    display(err_data)


def task3(df, df_mean):
    Y = df["percentageReds"].to_numpy()
    X = df[["rating"]].to_numpy()
    df_mean = df_mean[["rating", "percentageReds"]]

    # Number of folds
    L = 20

    # Create  L folds
    N = len(X)
    indices = np.random.choice(N, N, replace=False)
    X_folds = np.array(np.array_split(X[indices], L), dtype=object)
    Y_folds = np.array(np.array_split(Y[indices], L), dtype=object)

    # Linear Regression
    error = []
    for i in range(L):
        print(i / L * 100, "%")
        # Create training and test data
        X_train = np.concatenate(X_folds[np.arange(L) != i], axis=0)
        Y_train = np.concatenate(Y_folds[np.arange(L) != i], axis=0)
        X_test = X_folds[i]
        Y_test = Y_folds[i]

        # Compute error
        regression = LR.LinearRegression(df_mean)
        regression.train(X_train, Y_train)
        error.append(compute_error(regression, X_test, Y_test))
    error = np.mean(error)

    # Print error
    print("\nerror rate, linear regression:")
    print(error)

    color_rating_index = 0  # index of the color rating in df
    L = 20  # number of folds

    # Load csv-file where we save the mean squared errors
    err_data = pd.read_csv("errorsLie.txt", sep=",", index_col=False)

    # Load original data set
    Y = df["percentageReds"].to_numpy()
    X = df[["rating"]].to_numpy()

    # Linear Regression
    # Shuffle data
    Y_shuffled = Y
    X_shuffled = shuffle_data(X, color_rating_index)

    # create  L folds
    N = len(X_shuffled)
    indices = np.random.choice(N, N, replace=False)
    X_folds = np.array(np.array_split(X_shuffled[indices], L), dtype=object)
    Y_folds = np.array(np.array_split(Y_shuffled[indices], L), dtype=object)

    error = []
    for i in range(L):
        print(i / L * 100, "%")
        # Create training and test data
        X_train = np.concatenate(X_folds[np.arange(L) != i], axis=0)
        Y_train = np.concatenate(Y_folds[np.arange(L) != i], axis=0)
        X_test = X_folds[i]
        Y_test = Y_folds[i]

        # Compute error
        regression = LR.LinearRegression(df_mean)
        regression.train(X_train, Y_train)
        error.append(compute_error(regression, X_test, Y_test))
    error = np.mean(error)

    # Print error and save the value
    print("\nerror rate, linear regression:")
    print(error)

    err_data.loc[len(err_data)] = [error]

    err_data.to_csv("errorsLie.txt", sep=",", index=False)


def task4(df):
    # Compute covariance matrices
    Y = df["percentageReds"].to_numpy()
    X = df["weight"].to_numpy()
    print(np.cov(X, Y))
    Y = df["rating"].to_numpy()
    print(np.cov(X, Y), "\n")

    Y = df["percentageReds"].to_numpy()
    X = df["Position_Back"].to_numpy()
    print(np.cov(X, Y))
    Y = df["rating"].to_numpy()
    print(np.cov(X, Y))


def main():
    df, df_mean = DP.data_preparation()

    task1(df, df_mean)
    task2(df, df_mean)
    task3(df, df_mean)
    task4(df)


if __name__ == '__main__':
    main()
