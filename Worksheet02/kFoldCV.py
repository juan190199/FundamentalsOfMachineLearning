import numpy.testing as nt
import numpy as np
from random import randrange

class kFoldCV:
    """
    Provides train/test indices to split data in train/test sets.
    Split dataset into k consecutive folds;
    each fold is then used once as a validation while the k - 1 remaining folds form the training set

    Parameters
    -----------
    n_folds: int - number of folds. Must be at least 2
    shuffle: bool, default True - whether to shuffle the data before splitting into batches
    seed: int, default 4321 - When shuffle = True, pseudo-random number generator state used for shuffling;
    this ensures reproducibility
    """
    def __init__(self, n_folds, shuffle=True, seed=4321):
        self.seed = seed
        self.shuffle = shuffle
        self.n_folds = n_folds

    def split(self, dataset):
        """
        Create train/test split for k fold
        :param dataset: dataset
        :return:
        """
        # Shuffle modifies indices inplace
        n_samples = dataset.shape[0]
        indices = np.arange(n_samples)
        if self.shuffle:
            rstate = np.random.RandomState(self.seed)
            rstate.shuffle(indices)

        for test_mask in self._iter_test_masks(n_samples, indices):
            train_index = indices[np.logical_not(test_mask)]
            test_index = indices[test_mask]
            yield train_index, test_index

    def _iter_test_masks(self, n_samples, indices):
        """
        Create the mask for the test set, then the indices that are not in the test set belongs in the training set
        :param n_samples:
        :param indices:
        :return:
        """
        fold_sizes = (n_samples//self.n_folds) * np.ones(self.n_folds, dtype=np.int)
        fold_sizes[:n_samples % self.n_folds] += 1

        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            test_mask = np.zeros(n_samples, dtype=np.bool)
            test_mask[test_indices] = True
            yield test_mask
            current = stop
