from sklearn.datasets import load_digits
from sklearn import model_selection
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as nt
import pandas as pd

import Worksheet02.general as gen
import Worksheet02.kFoldCV as kfcv


def dist_loop(training, test):
    """
    Calculates distance between vectors in two matrices
    :param training: matrix of shape (n_instances, D). n_instances = number of instances in the training set.
    D = pixels per image
    :param test: matrix of shape (batch_size, D). batch_size = number of instances in the test set. D = pixels per image
    :return: matrix of shape (n_instances, M)
    """
    n_instances = training.shape[0]
    batch_size = test.shape[0]
    distances = np.empty(shape=(n_instances, batch_size))
    for i in range(n_instances):
        for j in range(batch_size):
            dist = gen.euclidean_distance(training[i, :], test[j, :], loop=2)
            distances[i, j] = dist
    return distances


def dist_vec_one_loop(training, test):
    """
    Calculates distance between vectors in two matrices
    :param training: matrix of shape (n_instances, D). n_instances = number of instances in the training set.
    D = pixels per image
    :param test: matrix of shape (batch_size, D). batch_size = number of instances in the test set. D = pixels per image
    :return: matrix of shape (n_instances, batch_size)
    """
    n_instances = training.shape[0]
    batch_size = test.shape[0]
    distances = np.empty(shape=(n_instances, batch_size))
    for i in range(batch_size):
        distances[:, i] = gen.euclidean_distance(training, test[i, :], loop=1)
    return distances


def dist_vec_v1(training, test):
    """
    Calculates distance between vectors in two matrices
    :param training: matrix of shape (n_instances, D). n_instances = number of instances in the training set.
    D = pixels per image
    :param test: matrix of shape (batch_size, D). batch_size = number of instances in the test set. D = pixels per image
    :return: matrix of shape (n_instances, M)
    """
    distances = gen.euclidean_distance(training, test, version='v1')
    return distances


def dist_vec_v2(training, test):
    """
    Calculates distance between vectors in two matrices
    :param training: matrix of shape (n_instances, D). n_instances = number of instances in the training set.
    D = pixels per image
    :param test: matrix of shape (batch_size, D). batch_size = number of instances in the test set. D = pixels per image
    :return: matrix of shape (n_instances, M)
    """
    distances = gen.euclidean_distance(training, test, version='v2')
    return distances


def dist_vec_original(training, test):
    distances = gen.euclidean_distance(training, test, version='original')
    return distances


def get_knn(k, training, test):
    """
    Get k nearest neighbors of each point in the test set
    :param k: number of neighbors to include in the majority vote
    :param training: matrix of shape (n_instances, D). n_instances = number of instances in the training set.
    D = pixels per image
    :param test: matrix of shape (batch_size, D). batch_size = number of instances in the test set. D = pixels per image
    :return: ndarray of dimenstion (k, batch_size) containing in each column the idx of knns' in training set.
    """
    dist = dist_vec_v1(training, test)
    # ndarray of dimenstion (k, batch_size) containing in each column the idx of knns' in training set.
    idx = np.argsort(dist, axis=0)[:k, :]
    return idx


def knn_classifier(k, training, labels, test, C=10):
    """
    Predicts class of a given test set
    :param C: Number of classes
    :param labels: labels of training data
    :param k: number of neighbors to include in the majority vote
    :param training: matrix of shape (n_instances, D). n_instances = number of instances in the training set.
    D = pixels per image
    :param test: matrix of shape (batch_size, D). batch_size = number of instances in the test set. D = pixels per image
    :return: ndarray of shape (batch_size, ) containing label prediction for each test point
    """
    idx = get_knn(k, training, test)
    # ndarray of dimension (k, batch_size) containing per column labels of the knn of the training set
    count = np.take(labels, idx)
    # ndarray of dimension (C, batch_size) containing votes for each class per test point
    election = np.apply_along_axis(lambda x: np.bincount(x, minlength=C), axis=0, arr=count)
    # ndarray of shape (batch_size, ) containing label prediction for each test point
    prediction = np.argmax(election, axis=0)
    return prediction


def calculate_error_knn_classifier(X_train, Y_train, X_test, Y_test, batch_size, K):
    """
    Calculates error by prediction with knn-classifier
    :param X_train: matrix of shape (n_instances, n_features)
    :param Y_train: matrix of shape (n_instances, 1)
    :param X_test: matrix of shape (batch_size, n_features)
    :param Y_test: matrix of shape (batch_size, 1)
    :param batch_size: integer - number of instances of test set
    :param K: array with number of neighbors to include in the majority vote
    :return: tuple - oses: error rates per k, errors: list(dictionary) - keys: "ose", "k"
    """
    oses = np.empty(len(K))
    i = 0
    errors = []
    for k in K:
        predictions = knn_classifier(k, X_train, Y_train, X_test)
        n_errors = np.count_nonzero(Y_test != predictions)
        ose = n_errors / batch_size
        oses[i] = ose
        errors.append({
            "k": k,
            "ose": ose
        })
        i += 1
    return oses, errors


def main():
    # Task 1
    digits = load_digits()

    print(digits.keys())

    data = digits['data']
    images = digits['images']
    target = digits['target']
    target_names = digits['target_names']

    print(digits['DESCR'])
    print(f"data.shape          = {data.shape}")
    print(f"data.dtype          = {data.dtype}")
    print(f"images.shape        = {images.shape}")
    print(f"images.shape        = {images.dtype}")
    print(f"target.shape        = {target.shape}")
    print(f"target.shape        = {target.dtype}")
    print(f"target_names.shape  = {target_names.shape}")
    print(f"target_names.shape  = {target_names.dtype}")
    print(f"target[:20]         = {target[:20]}")
    """
    The digits dataset consists of 8x8 pixel images of digits. 
    The ``images`` attribute of the dataset stores 8x8 arrays of grayscale values for each image. 
    We will use these arrays to visualize the first 10 images. The ``target`` attribute of the dataset stores the digit 
    each image represents and this is included in the title of the 10 plots below.
    """

    """
    For scientific analysis, it's important that we can see the exact value of each pixel in an image array, 
    even if interpolation = 'bicubic' often yields visually more pleasing results.
    """
    # Plot images form 0 to 9
    _, axes = plt.subplots(2, 5)
    for ax, image, label in zip(axes[0, :], images[:5], target[:5]):
        ax.set_axis_off()
        assert 2 == len(image.shape)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: %i' % label)
    for ax, image, label in zip(axes[1, :], images[5:], target[5:]):
        ax.set_axis_off()
        assert 2 == len(image.shape)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='bicubic')
        ax.set_title('Training: %i' % label)
    plt.show()

    # Split of training and test data using sklearn
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(
        digits.data,
        digits.target,
        test_size=0.4,
        random_state=0
    )

    # Task 2 & 3
    # distance_loop = dist_loop(X_train, X_test)
    distance_vec_one_loop = dist_vec_one_loop(X_train, X_test)
    distance_vec_v1 = dist_vec_v1(X_train, X_test)
    distance_vec_v2 = dist_vec_v2(X_train, X_test)
    distance_vec_original = dist_vec_original(X_train, X_test)
    # nt.assert_array_equal(distance_loop, distance_vec_v1)
    nt.assert_array_equal(distance_vec_one_loop, distance_vec_v1)
    nt.assert_array_equal(distance_vec_v1, distance_vec_v2)
    nt.assert_array_equal(distance_vec_v1, distance_vec_original)

    # Task 4
    predictions = knn_classifier(7, X_train, Y_train, X_test)

    # Filter data
    filt_Y_train_idx = np.argwhere((Y_train == 3) | (Y_train == 9)).flatten()
    filt_X_train = X_train[filt_Y_train_idx, :]
    filt_Y_train = Y_train[filt_Y_train_idx]

    filt_Y_test_idx = np.argwhere((Y_test == 3) | (Y_test == 9)).flatten()
    filt_X_test = X_test[filt_Y_test_idx]
    filt_Y_test = Y_test[filt_Y_test_idx]

    # Create filtered images
    img_train = filt_X_train.reshape(len(filt_Y_train_idx), 8, 8)
    img_test = filt_X_test.reshape(len(filt_Y_test_idx), 8, 8)

    # Plot filtered images
    _, axes = plt.subplots(2, 5)
    for ax, image, label in zip(axes[0, :], img_train[:5], filt_Y_train[:5]):
        ax.set_axis_off()
        assert 2 == len(image.shape)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Training: %i' % label)
    for ax, image, label in zip(axes[1, :], img_test[5:], filt_Y_test[5:]):
        ax.set_axis_off()
        assert 2 == len(image.shape)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='bicubic')
        ax.set_title('Training: %i' % label)
    plt.show()

    # predictions = knn_classifier(7, filt_X_train, filt_Y_train, filt_X_test)
    # Prediction and error rates for different number of neighbors to include in the majority vote for filtered data
    K = [1, 3, 5, 9, 17, 33]
    batch_size = len(filt_Y_test)
    _, filt_errors = calculate_error_knn_classifier(
        X_train=filt_X_train,
        Y_train=filt_Y_train,
        X_test=filt_X_test,
        Y_test=filt_Y_test,
        batch_size=batch_size,
        K=K
    )

    # Dataframe reporting error rates per k
    print("\n Error rates for class pairs (3, 9)")
    df = pd.DataFrame(filt_errors)
    df = df.groupby(["k"]).first()
    print(df)

    # Convert lis of dictionaries containing errors to one dimensional list of errors
    list_filt_errors = np.empty(len(K))
    for i in range(len(filt_errors)):
        list_filt_errors[i] = filt_errors[i]['ose']

    # Prediction and error rates for different number of neighbors to include in the majority vote for raw data
    _, errors = calculate_error_knn_classifier(
        X_train=X_train,
        Y_train=Y_train,
        X_test=X_test,
        Y_test=Y_test,
        batch_size=batch_size,
        K=K
    )

    # Dataframe reporting error rates per k
    print("\n Error rates for all classes")
    df = pd.DataFrame(errors)
    df = df.groupby(["k"]).first()
    print(df)

    # Convert lis of dictionaries containing errors to one dimensional list of errors
    list_errors = np.empty(len(K))
    for i in range(len(filt_errors)):
        list_errors[i] = errors[i]['ose']

    # Plot average error rate for class pair (3, 9)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].set_title("Average error rate for class pair (3,9)")
    axes[0].plot(K, list_filt_errors)
    axes[0].scatter(K, list_filt_errors, c="Red")
    axes[0].set_ylabel("average error in %")
    axes[0].set_xlabel("number of neighbors k")

    axes[1].set_title("Average error rate for all classes")
    axes[1].plot(K, list_errors)
    axes[1].scatter(K, list_errors, c="Red")
    axes[1].set_ylabel("average error in %")
    axes[1].set_xlabel("number of neighbors k")

    fig.tight_layout()
    plt.show()

    # Task 5
    X = np.vstack((filt_X_train, filt_X_test))
    Y = np.hstack((filt_Y_train, filt_Y_test))
    arr_n_folds = [2, 5, 10]
    oses = np.empty(shape=(len(arr_n_folds), len(K)))
    i = 0
    for n_folds in arr_n_folds:
        kf = kfcv.kFoldCV(n_folds=n_folds, shuffle=True, seed=4321)
        for train_index, test_index in kf.split(dataset=X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            _, errors = calculate_error_knn_classifier(
                X_train=X_train,
                Y_train=Y_train,
                X_test=X_test,
                Y_test=Y_test,
                batch_size=batch_size,
                K=K
            )
            oses[i, :] = list(map(itemgetter('ose'), errors))
        i += 1

    errors_dict = {
        "k": K,
        "Mean": np.mean(oses, axis=0),
        "Std": np.std(oses, axis=0)
    }

    # Dataframe reporting error rates per k (k-folds cross validation)
    df = pd.DataFrame(errors_dict)
    df.groupby(["k"]).first()
    print(df)

    # knn - Sklearn
    oses = np.empty(shape=(len(arr_n_folds), len(K)))
    j = 0
    for n_folds in arr_n_folds:
        kf = kfcv.kFoldCV(n_folds=n_folds, shuffle=True, seed=4321)
        for train_index, test_index in kf.split(dataset=X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            oses_k = np.empty(len(K))
            i = 0
            errors = []
            for k in K:
                model = KNeighborsClassifier(k)
                model.fit(X=X_train, y=Y_train)
                prediction = model.predict(X=X_test)
                n_errors = np.count_nonzero(Y_test != prediction)
                ose = n_errors / batch_size
                oses_k[i] = ose
                errors.append({
                    "k": k,
                    "ose": ose
                })
                i += 1

            oses[j, :] = list(map(itemgetter('ose'), errors))
        j += 1

    errors_dict = {
        "k": K,
        "Mean": np.mean(oses, axis=0),
        "Std": np.std(oses, axis=0)
    }

    # Dataframe reporting error rates per k (Sklearn, k-folds cross validation)
    df = pd.DataFrame(errors_dict)
    df.groupby(["k"]).first()
    print(df)


if __name__ == '__main__':
    main()
