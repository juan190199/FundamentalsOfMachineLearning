import numpy as np
import numpy.testing as nt


def euclidean_distance(x, y, loop=None, version=None):
    if loop == 2:
        distance = np.sqrt(np.sum(np.square(np.subtract(x, y))))
        nt.assert_array_equal(distance, np.linalg.norm(np.subtract(x, y)))
        return distance
    if loop == 1:
        distance = np.sqrt(np.sum(np.square(np.subtract(x, y)), axis=1))
        nt.assert_array_equal(distance, np.linalg.norm(np.subtract(x, y), axis=1))
        return distance
    if version == 'v1':
        distance = np.sqrt(
            np.sum(
                np.square(np.subtract(x[:, np.newaxis], y)), axis=2))
        return distance
    if version == 'v2':
        distance = np.sqrt(
            np.sum(
                np.square(np.expand_dims(x, axis=1) - np.expand_dims(y, axis=0)),
                axis=2,
            )
        )
        return distance
    if version == 'original':
        distance = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)
        return distance

