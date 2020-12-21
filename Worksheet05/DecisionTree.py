import numpy as np
import Worksheet05.BaseClasses as BaseClasses


class DecisionTree(BaseClasses.Tree):
    def __init__(self):
        super(DecisionTree, self).__init__()

    def train(self, data, labels, n_min=20):
        '''
        data: the feature matrix for all digits
        labels: the corresponding ground-truth responses
        n_min: termination criterion (don't split if a node contains fewer instances)
        '''
        N, D = data.shape
        D_try = int(np.sqrt(D))  # how many features to consider for each split decision

        # initialize the root node
        self.root.data = data
        self.root.labels = labels

        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0]  # number of instances in present node
            if n >= n_min and not node_is_pure(node):
                # Call 'make_decision_split_node()' with 'D_try' randomly selected
                # feature indices. This turns 'node' into a split node
                # and returns the two children, which must be placed on the 'stack'.
                ...  # your code here
            else:
                # Call 'make_decision_leaf_node()' to turn 'node' into a leaf node.
                ...  # your code here

    def predict(self, x):
        leaf = self.find_leaf(x)
        # compute p(y | x)
        return ...  # your code here


def make_decision_split_node(node, feature_indices):
    '''
    node: the node to be split
    feature_indices: a numpy array of length 'D_try', containing the feature
                     indices to be considered in the present split
    '''
    n, D = node.data.shape

    # find best feature j (among 'feature_indices') and best threshold t for the split
    ...  # your code here

    # create children
    left = BaseClasses.Node()
    right = BaseClasses.Node()

    # initialize 'left' and 'right' with the data subsets and labels
    # according to the optimal split found above
    left.data = ...  # data in left node
    left.labels = ...  # corresponding labels
    right.data = ...
    right.labels = ...

    # turn the current 'node' into a split node
    # (store children and split condition)
    node.left = left
    node.right = right
    node.feature = ...
    node.threshold = ...

    # return the children (to be placed on the stack)
    return left, right


def make_decision_leaf_node(node):
    '''
    node: the node to become a leaf
    '''
    # compute and store leaf response
    node.N = ...
    node.response = ... # your code here


def node_is_pure(node):
    '''
    check if 'node' ontains only instances of the same digit
    '''
    return ... # your code here


