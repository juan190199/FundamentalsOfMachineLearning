import Worksheet05.DecisionTree as DecisionTree

import numpy as np


class DecisionForest:
    def __init__(self, n_trees):
        # Create ensemble
        self.trees = [DecisionTree.DecisionTree() for i in range(n_trees)]

    def train(self, data, labels, n_min=0):
        for tree in self.trees:
            # Train each tree, using a bootstrap sample of the data
            bootstrap = np.random.choice(len(data), size=len(data))
            tree.train(data[bootstrap], labels[bootstrap], n_min)

    def predict(self, x):
        # Compute the ensemble prediction
        return np.mean([tree.predict(x) for tree in self.trees], axis=0)
