import DecisionTree


class DecisionForest():
    def __init__(self, n_trees):
        # create ensemble
        self.trees = [DecisionTree.DecisionTree() for i in range(n_trees)]

    def train(self, data, labels, n_min=0):
        for tree in self.trees:
            # train each tree, using a bootstrap sample of the data
            ... # your code here

    def predict(self, x):
        # compute the ensemble prediction
        return ... # your code here
