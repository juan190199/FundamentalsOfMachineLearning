import numpy as np
from sklearn.model_selection import train_test_split


def data_preparation(digits, test_percentage=0.33, random_seed=0):
    """
    Filter the digits (1, 7) from the data set and randomly splits it in train and test set.
    :param digits:
    :param test_percentage:
    :param random_seed:
    :return:
    """
    data = digits['data']
    target = digits['target']

    # Data filtering
    num_1, num_2 = 1, 7
    mask = np.logical_or(target == num_1, target == num_2)
    data = data[mask] / data.max()
    target = target[mask]

    # Relabel targets
    target[target == num_1] = 0
    target[target == num_2] = 1

    # Random split
    X_train, X_test, Y_train, Y_test = train_test_split(
        data,
        target,
        test_size=test_percentage,
        random_state=random_seed
    )

    return X_train, X_test, Y_train, Y_test


def distance_from_mean(x, mean):
    """
    Computes L2 distance to the mean
    :param x:
    :param mean:
    :return:
    """
    return np.sqrt(np.sum((x - mean)**2, axis=-1))


def nearest_mean(training_features, training_labels, test_features):
    """
    This function returns the nearest mean predictions given input training_features, training_labels, and test_features
    :param training_features: ndarray: (N_training, 2)
    :param training_labels: ndarray: (N_training, 1)
    :param test_features: ndarray: (N_test, 2)
    :return: test_predictions: ndarray: (N_test, 1)
    """

    classes_list = np.unique(training_labels)
    mean_points = []
    # Find all mean points
    for label in classes_list:
        mean_points.append(np.mean(training_features[training_labels == label], axis=0))

    distance2mean = np.zeros((test_features.shape[0],
                              classes_list.shape[0]))
    # Compute the distances between test_features and all mean
    for label in classes_list:
        distance2mean[:, label] = distance_from_mean(test_features, mean_points[label])

    return np.argmin(distance2mean, axis=-1), mean_points

