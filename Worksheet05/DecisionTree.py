import numpy as np

import Worksheet05.BaseClasses as BaseClasses


class DecisionTree(BaseClasses.Tree):
    def __init__(self):
        super(DecisionTree, self).__init__()

    def train(self, data, labels, n_min=20):
        """
        :param data: the feature matrix for all digits
        :param labels: the corresponding ground-truth responses
        :param n_min: termination criterion (don't split if a node contains fewer instances)
        :return:
        """
        N, D = data.shape
        D_try = int(np.sqrt(D))  # How many features to consider for each split decision

        # Initialize the root node
        self.root.data = data
        self.root.labels = labels

        # Put root in stack
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0]  # Number of instances in present node
            if n >= n_min and not node_is_pure(node):
                # Call 'make_decision_split_node()' with 'D_try' randomly selected
                # feature indices. This turns 'node' into a split node
                # and returns the two children, which must be placed on the 'stack'.
                perm = np.random.permutation(D)  # Permutate D indices
                # Select :D_try of permuted indices for 'make_decision_split_node()'
                left, right = make_decision_split_node(node, perm[:D_try])
                # Put children in stack
                stack.append(left)
                stack.append(right)
            else:
                # Call 'make_decision_leaf_node()' to turn 'node' into a leaf node.
                make_decision_leaf_node(node)

    def predict(self, x):
        leaf = self.find_leaf(x)
        # compute p(y | x)
        return leaf.response


def make_decision_split_node(node, feature_indices):
    """
    :param node: the node to be split
    :param feature_indices: a numpy array of length 'D_try', containing the feature
    indices to be considered in the present split
    :return:
    """
    n, D = node.data.shape

    # Find best feature j (among 'feature_indices') and best threshold t for the split
    e_min = 1e100
    j_min, t_min = 0, 0
    for j in feature_indices:
        # Remove duplicate features
        dj = np.sort(np.unique(node.data[:, j]))
        # Compute candidate thresholds in the middle between consecutive feature values
        tj = (dj[1:] + dj[:-1]) / 2
        # Gini impurities of the resulting children node have to be computed for each candidate threshold
        for t in tj:
            left_indices = node.data[:, j] <= t
            nl = np.sum(left_indices)
            ll = node.labels[left_indices]
            el = nl * (1 - np.sum(np.square(np.bincount(ll) / nl)))
            nr = n - nl
            lr = node.labels[node.data[:, j] > t]
            er = nr * (1 - np.sum(np.square(np.bincount(lr) / nr)))

            # Choose the the best threshold that minimizes sum of Gini impurities
            if el + er < e_min:
                e_min = el + er
                j_min = j
                t_min = t

    # Create children
    left = BaseClasses.Node()
    right = BaseClasses.Node()

    # Initialize 'left' and 'right' with the data subsets and labels according to the optimal split found above
    left.data = node.data[node.data[:, j_min] <= t_min, :]
    left.labels = node.labels[node.data[:, j_min] <= t_min]
    right.data = node.data[node.data[:, j_min] > t_min, :]
    right.labels = node.labels[node.data[:, j_min] > t_min]

    # Turn the current 'node' into a split node (store children and split condition)
    node.left = left
    node.right = right
    node.feature = j_min
    node.threshold = t_min

    return left, right


def make_decision_leaf_node(node):
    """

    :param node: The node to become a leaf
    :return:
    """
    node.N = node.labels.shape[0]
    node.response = np.bincount(node.labels, minlength=10) / node.N


def node_is_pure(node):
    """
    Check if 'node' ontains only instances of the same digit
    :param node:
    :return: boolean variable
    """
    return np.unique(node.labels).shape[0] == 1


