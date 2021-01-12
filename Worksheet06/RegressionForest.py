from Worksheet06 import RegressionTree as RT

import numpy as np


class RegressionForest:
    def __init__(self, n_trees, df_mean):
        # create ensemble
        self.trees = [RT.RegressionTree(df_mean) for i in range(n_trees)]

    def train(self, data, labels, n_min=1000):
        for tree in self.trees:
            # train each tree, using a bootstrap sample of the data
            bootstrap_indices = np.random.choice(len(labels), len(labels))
            bootstrap_data = np.array([data[i] for i in bootstrap_indices])
            bootstrap_labels = np.array([labels[i] for i in bootstrap_indices])
            tree.train(bootstrap_data, bootstrap_labels, n_min=n_min)

    def predict(self, x):
        predictions = np.array([])
        for tree in self.trees:
            predictions = np.append(predictions, tree.predict(x))
        return np.mean(predictions)

    def merge(self, forest):
        self.trees = self.trees + forest.trees
