from sklearn.datasets import load_digits
from sklearn import model_selection
import numpy as np
import matplotlib.pyplot as plt

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
            dist = gen.euclidean_distance(training[i, :], test[j, :])
            distances[i][j]
    return distances


def dist_vec(training, test):
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
        dist = np.apply_along_axis(gen.euclidean_distance, axis=0, args=(training[i, :], test))


def main():
    digits = load_digits()

    print(digits.keys())

    data = digits['data']
    images = digits['images']
    target = digits['target']
    target_names = digits['target_names']

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
    plt.show()

    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(digits.data,
                                                                        digits.target,
                                                                        test_size=0.4,
                                                                        random_state=0)

    distance_loop = dist_loop(X_train, X_test)
    distance_vec = dist_vec(X_train, X_test)
    # assert distance_loop == distance_vec


if __name__ == '__main__':
    main()
