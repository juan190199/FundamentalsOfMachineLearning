from sklearn.datasets import load_digits
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt
import numpy.testing as nt

import general as gen


def dist_loop(training, test):
    """

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
            distances[i][j] = dist
    return distances


def dist_vec_one_loop(training, test):
    """

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


def dist_vec(training, test):
    """

    :param training: matrix of shape (n_instances, D). n_instances = number of instances in the training set.
    D = pixels per image
    :param test: matrix of shape (batch_size, D). batch_size = number of instances in the test set. D = pixels per image
    :return: matrix of shape (n_instances, M)
    """
    distances = gen.euclidean_distance(training, test)
    return distances


def main():
    digits = load_digits()

    print(digits.keys())

    data = digits['data']
    images = digits['images']
    target = digits['target']
    target_names = digits['target_names']

    print(digits['DESCR'])
    print(data.dtype)
    print(data.shape)
    """
    The digits dataset consists of 8x8 pixel images of digits. 
    The ``images`` attribute of the dataset stores 8x8 arrays of grayscale values for each image. 
    We will use these arrays to visualize the first 10 images. The ``target`` attribute of the dataset stores the digit 
    each image represents and this is included in the title of the 10 plots below.
    """

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
    # plt.show()

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(digits.data,
                                                                        digits.target,
                                                                        test_size=0.4,
                                                                        random_state=0)

    # distance_loop = dist_loop(X_train, X_test)
    distance_vec_one_loop = dist_vec_one_loop(X_train, X_test)
    distance_vec = dist_vec(X_train, X_test)
    nt.assert_array_equal(distance_vec_one_loop, distance_vec)


if __name__ == '__main__':
    main()
