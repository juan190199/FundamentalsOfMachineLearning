import DensityTree


class DensityForest():
    def __init__(self, n_trees):
        # create ensemble
        self.trees = [DensityTree.DensityTree() for i in range(n_trees)]

    def train(self, data, prior, n_min=20):
        for tree in self.trees:
            # train each tree, using a bootstrap sample of the data
            ... # your code here

    def predict(self, x):
        # compute the ensemble prediction
        return ... # your code here
