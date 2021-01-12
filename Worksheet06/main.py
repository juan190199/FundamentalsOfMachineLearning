import Worksheet06.Data.DataPreparation as DP
import Worksheet06.LinearRegression as LR
import Worksheet06.RegressionForest as RF

import numpy as np


def compute_error(model, test_features, test_labels):
    mean_squared_error = 0
    n = len(test_features)

    for i in range(n):
        mean_squared_error = mean_squared_error + (test_labels[i] - model.predict(test_features[i])) ** 2

    return mean_squared_error / n


def main():
    df, df_mean = DP.data_preparation()

    Y = df["percentageReds"].to_numpy()
    X = df.drop(labels=["percentageReds"], axis=1).to_numpy()

    # number of folds
    L = 10

    # create  L folds
    N = len(X)
    indices = np.random.choice(N, N, replace=False)
    X_folds = np.array(np.array_split(X[indices], L), dtype=object)
    Y_folds = np.array(np.array_split(Y[indices], L), dtype=object)

    # Error Linear Regression
    error = []
    for i in range(L):
        print(i / L * 100, "%")
        # create training and test data
        X_train = np.concatenate(X_folds[np.arange(L) != i], axis=0)
        Y_train = np.concatenate(Y_folds[np.arange(L) != i], axis=0)
        X_test = X_folds[i]
        Y_test = Y_folds[i]

        # compute error
        regression = LR.LinearRegression(df_mean)
        regression.train(X_train, Y_train)
        error.append(compute_error(regression, X_test, Y_test))
    error = np.mean(error)

    # print error
    print("\nError rate, linear regression:")
    print(error)

    # Error Regression Forest
    error = []
    for i in range(L):
        print(i / L * 100, "%")
        # create training and test data
        X_train = np.concatenate(X_folds[np.arange(L) != i], axis=0)
        Y_train = np.concatenate(Y_folds[np.arange(L) != i], axis=0)
        X_test = X_folds[i]
        Y_test = Y_folds[i]

        # compute error
        forest = RF.RegressionForest(5, df_mean)
        forest.train(X_train, Y_train, n_min=500)
        error.append(compute_error(forest, X_test, Y_test))
    error = np.mean(error)

    # print error
    print("\nerror rate, regression forest:")
    print(error)


if __name__ == '__main__':
    main()
