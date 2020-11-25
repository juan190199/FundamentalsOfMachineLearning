import numpy as np
import numpy.testing as nt


def euclidean_distance(x, y, loop=None):
    if loop == 2:
        distance = np.sqrt(np.sum(np.square(np.subtract(x, y))))
        nt.assert_array_equal(distance, np.linalg.norm(np.subtract(x, y)))
        return distance
    if loop == 1:
        distance = np.sqrt(np.sum(np.square(np.subtract(x, y)), axis=1))
        nt.assert_array_equal(distance, np.linalg.norm(np.subtract(x, y), axis=1))
        return distance
    else:
        distance = np.sqrt(np.sum(np.square(np.subtract(x[:, np.newaxis], y)), axis=2))
        return distance
