import Worksheet05.DensityForest as DensityForest

import numpy as np


class GenerativeClassifierDensityForest:
    def __init__(self, n_trees):
        # create 10 instances of DensiryForest
        self.forests = [DensityForest.DensityForest(n_trees) for i in range(10)]

    def train(self, data, target, n_min=20):
        # train 10 forests, each with one subset of digits
        data_subsets = [data[target == i] for i in range(10)]
        N = len(target)
        for i, forest in enumerate(self.forests):
            forest.train(data_subsets[i], len(data_subsets[i]) / N, n_min)

    def predict(self, x):
        # return the digit y that maximizes p(y | x)
        return np.argmax([forest.predict(x) for forest in self.forests])
