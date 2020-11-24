import numpy as np


def euclidean_distance(x, y):
    distance = np.sqrt(np.sum(np.square(np.subtract(x, y))))
    assert distance == np.linalg.norm(np.subtract(x, y))
    return distance
