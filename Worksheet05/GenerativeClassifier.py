import numpy as np

import Worksheet05.DensityTree as DensityTree


class GenerativeClassifier:
    """
    This class combines 10 instances of DensityTree.
    One instance of DensityTree is to be trained with only one class of data from the training data.
    Therefore training the generative classifier with full dataset will first separate the data into 10 subsets
    (each for one digit) and then trains 10 DensityTrees, each trained for a digit subset.
    For prediction, a datapoint is be predicted by all 10 DensityTrees and each DensityTree returns p(x|y) * p(y)
    using predict of DensityTree class.
    The digit for the DensityTree maximizing p(x|y) * p(y) is the prediction of the generative classifier.
    """
    def __init__(self):
        # Create 10 instances of Density tree
        self.trees = [DensityTree.DensityTree() for i in range(10)]

    def train(self, data, target, n_min=20):
        # Train 10 trees, each with one subset of digits
        data_subsets = [data[target == i] for i in range(10)]
        N = len(target)
        for i, tree in enumerate(self.trees):
            tree.train(data_subsets[i], len(data_subsets[i]) / N, n_min)

    def predict(self, x):
        # Return the digit for the DensityTree that maximizes p(x | y) * p(y)
        return np.argmax([tree.predict(x) for tree in self.trees])

