import Worksheet05.DensityTree as DensityTree

import numpy as np


class DensityForest:
    def __init__(self, n_trees):
        # Create ensemble
        self.trees = [DensityTree.DensityTree() for i in range(n_trees)]

    def train(self, data, prior, n_min=20):
        for tree in self.trees:
            bootstrap_data = data[np.random.choice(len(data), size=len(data))]
            tree.train(bootstrap_data, prior, n_min)

    def predict(self, x):
        # Compute the ensemble prediction
        return np.mean([tree.predict(x) for tree in self.trees])
