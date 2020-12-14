# Data
from sklearn.datasets import load_digits

# Numbers
import numpy as np

import Worksheet03.general as gen
import Worksheet03.NearestMeanClassifier as NMC
import Worksheet03.QDA as QDA
import Worksheet03.LDA as LDA


def task3(xr_training, y_training, xr_test, y_test):
    # Find nearest mean predictions
    predicted_labels, mean_points = NMC.nearest_mean(xr_training, y_training, xr_test)

    # Print accuracy
    print("Accuracy Nearest Mean: ", np.mean(predicted_labels == y_test))

    NMC.visualization_NearestMean(xr_training, y_training, xr_test, y_test)


def task4(digits, xr_training, y_training, xr_test, y_test):
    # Fit QDA
    mu, cov, p = QDA.fit_qda(xr_training, y_training)

    # QDA predictions
    predicted_labels = QDA.predict_qda(mu, cov, p, xr_test)

    # Print accuracy
    print("Accuracy QDA: ", np.mean(predicted_labels == y_test))

    _, mean_points = NMC.nearest_mean(xr_training, y_training, xr_test)

    QDA.visualization_QDA(xr_training, y_training, xr_test, y_test, mu, cov, p, mean_points, simple=True)
    QDA.visualization_QDA(xr_training, y_training, xr_test, y_test, mu, cov, p, mean_points, simple=False)

    # Print accuracy with cross validation
    gen.cross_validation(digits, QDA.fit_qda, QDA.predict_qda, 100)


def task5(digits, xr_training, y_training, xr_test, y_test):
    # Fit LDA
    mu, cov, p = LDA.fit_lda(xr_training, y_training)

    # Find QDA Predictions
    predicted_labels = LDA.predict_lda(mu, cov, p, xr_test)

    # Print accuracy
    print("Accuracy LDA: ", np.mean(predicted_labels == y_test))

    _, mean_points = NMC.nearest_mean(xr_training, y_training, xr_test)

    LDA.visualization_LDA(xr_training, y_training, xr_test, y_test, mu, cov, p, mean_points, simple=True)
    LDA.visualization_LDA(xr_training, y_training, xr_test, y_test, mu, cov, p, mean_points, simple=False)

    # Print accuracy with cross validation
    gen.cross_validation(digits, LDA.fit_lda, LDA.predict_lda, 100)


def main():
    # Load data
    digits = load_digits()

    # Filtering data
    x_training, x_test, y_training, y_test = gen.data_preparation(digits, 0.33, 0)

    # Dimension reduction
    xr_training, xr_test = gen.reduce_dim(x_training), gen.reduce_dim(x_test)

    # Scatter plot
    gen.scatter_plot_simple(xr_training, y_training, "Training")
    gen.scatter_plot_simple(xr_test, y_test, "Test")

    # Dimension reduction
    _xr_training, _xr_test = gen.worse_reduce_dim(x_training), gen.worse_reduce_dim(x_test)

    # Scatter Plot
    gen.scatter_plot_simple(_xr_training, y_training, "Training - Worse features")

    # Dimension reduction
    xr_test = gen.reduce_dim(x_test)

    task3(xr_training, y_training, xr_test, y_test)

    task4(digits, xr_training, y_training, xr_test, y_test)

    task5(digits, xr_training, y_training, xr_test, y_test)


if __name__ == '__main__':
    main()
