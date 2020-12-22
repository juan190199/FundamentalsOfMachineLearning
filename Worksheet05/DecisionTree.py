import numpy as np
import copy
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
                rd_indices = np.random.choice(np.arange(0, n), D_try, replace=True)
                left, right = make_decision_split_node(node, N, rd_indices)
                stack.append(left)
                stack.append(right)
            else:
                # Call 'make_decision_leaf_node()' to turn 'node' into a leaf node.
                make_decision_leaf_node(node)

    def predict(self, x):
        leaf = self.find_leaf(x)
        # compute p(y | x)
        return ...  # your code here


def make_decision_split_node(node, feature_indices):
    """

    :param node: the node to be split
    :param feature_indices: a numpy array of length 'D_try', containing the feature
    indices to be considered in the present split
    :return:
    """
    n, D = node.data.shape

    # find best feature j (among 'feature_indices') and best threshold t for the split
    for j in feature_indices:
        data_unique = np.uique(node.data[:, j])
        tj = (data_unique[1:] + data_unique[:-1]) / 2
        for t in tj:
            # Calculate volume children nodes
            m_left, M_right = copy.deepcopy(m), copy.deepcopy(M)
            M_left = copy.deepcopy(M)
            M_left[j] = t
            m_right = copy.deepcopy(m)
            m_right[j] = t

            # Calculate volume left and right children
            vol_left = np.prod(M_left - m_left)
            vol_right = np.prod(M_right - m_right)
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


