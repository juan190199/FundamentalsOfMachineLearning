# Numbers
import numpy as np

# Data
from sklearn.datasets import load_digits

# Plot
import matplotlib.pyplot as plt

from Worksheet03.general import data_preparation
from Worksheet03.QDA import fit_qda


def data_generation(mu, cov, num_new_instances=10, newshape=(8, 8)):
    """
    Generates num_new_instances per label
    :param mu: ndarray: (num_labels, )
    :param cov: ndarray: (num_labels, num_features, num_features)
    :param num_new_instances: int
    :param newshape: tuple: specifies image shape
    :return: gen_data, labels
    """
    num_labels = len(mu)
    gen_data = np.empty(shape=(10 * num_labels, newshape[0], newshape[1]))
    labels = np.empty(shape=(10 * num_labels), dtype=int)

    counter = 0
    label = 0
    for i in range(num_labels * num_new_instances):
        if i % 10 == 0:
            if i == 0:
                pass
            else:
                label += 1
        labels[counter] = label
        gen_data[counter, :, :] = np.random.multivariate_normal(mu[label], cov[label]).reshape(newshape)
        counter += 1

    return gen_data, labels


def plot_data(images, target, num_new_instances=10):
    _, axes = plt.subplots(2, int(num_new_instances/2))
    target = [target] * num_new_instances
    for ax, image, label in zip(axes[0, :], images[:int(num_new_instances / 2)], target[:int(num_new_instances/2)]):
        ax.set_axis_off()
        assert 2 == len(image.shape)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title('Label: %i' % label, fontsize='x-small')
    for ax, image, label in zip(axes[1, :], images[int(num_new_instances / 2):], target[int(num_new_instances/2):]):
        ax.set_axis_off()
        assert 2 == len(image.shape)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation='bicubic')
        ax.set_title('Label: %i' % label, fontsize='x-small')
    plt.show()


def main():
    # Load data
    digits = load_digits()

    # Filtering data
    x_training, _, y_training, _ = data_preparation(digits, 0.33, 0)

    # Fit QDA
    mu, cov, p = fit_qda(x_training, y_training)

    num_new_instances = 10
    gen_data, labels = data_generation(mu, cov, num_new_instances=num_new_instances)

    # Plot new instances label i
    for label in np.unique(labels):
        images = gen_data[labels == label]
        plot_data(images, label, num_new_instances=num_new_instances)


if __name__ == '__main__':
    main()
