import numpy as np
import copy
from sklearn.datasets import load_digits

import Worksheet05.BaseClasses as BaseClasses
import Worksheet05.general as gen


class DensityTree(BaseClasses.Tree):
    def __init__(self):
        super(DensityTree, self).__init__()

    def train(self, data, prior, n_min=20):
        """

        :param data: the feature matrix for the digit under consideration
        :param prior: the prior probability of this digit
        :param n_min: termination criterion (don't split if a node contains fewer instances)
        :return:
        """
        self.prior = prior
        N, D = data.shape
        D_try = int(np.sqrt(D))  # number of features to consider for each split decision

        # find and remember the tree's bounding box,
        # i.e. the lower and upper limits of the training feature set
        m, M = np.min(data, axis=0), np.max(data, axis=0)
        self.box = m.copy(), M.copy()

        # identify invalid features and adjust the bounding box
        # (If m[j] == M[j] for some j, the bounding box has zero volume,
        #  causing divide-by-zero errors later on. We must exclude these
        #  features from splitting and adjust the bounding box limits
        #  such that invalid features have no effect on the volume.)
        valid_features = np.where(m != M)[0]
        invalid_features = np.where(m == M)[0]
        M[invalid_features] = m[invalid_features] + 1

        # initialize the root node
        self.root.data = data
        self.root.box = m.copy(), M.copy()

        # build the tree
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0]  # number of instances in present node
            if n >= n_min:
                # Call 'make_density_split_node()' with 'D_try' randomly selected
                # indices from 'valid_features'. This turns 'node' into a split node
                # and returns the two children, which must be placed on the 'stack'.
                # your code here
                n_valid_feat = len(valid_features)
                rd_indices = np.random.choice(np.arange(0, n_valid_feat), D_try, replace=True)
                rd_valid_feat = data[:, valid_features[rd_indices]]

                left, right = make_density_split_node(node, N, rd_indices)
                stack.push(left)
                stack.push(right)

            else:
                # Call 'make_density_leaf_node()' to turn 'node' into a leaf node.
                make_density_leaf_node(node, n)

    def predict(self, x):
        leaf = self.find_leaf(x)
        # ToDo: return prior and likelihood
        # return p(x | y) * p(y) if x is within the tree's bounding box
        # and return 0 otherwise
        if x in leaf:
            return self.prior * ...
        else:
            return 0


def make_density_split_node(node, N, feature_indices):
    """

    :param node: the node to be split
    :param N: the total number of training instances for the current class
    :param feature_indices: a numpy array of length 'D_try', containing the feature
    indices to be considered in the present split
    :return:
    """
    n, D = node.data.shape
    m, M = node.box

    # find best feature j (among 'feature_indices') and best threshold t for the split
    e_min = float("inf")
    j_min, t_min = None, None
    volume_node = np.prod(M - m)  # Used for assert. (vol_left + vol_right = volume_node)

    for j in feature_indices:
        # For each feature considered, remove duplicate feature values
        # Duplicate feature values have to be removed since there is no possible threshold
        # in between features with the same values
        # (The midpoint of two features with the same value would be the same feature value)
        data_unique = np.unique(node.data[:, j])
        # Compute candidate thresholds
        tj = (data_unique[1:] + data_unique[:-1]) / 2

        # ToDo: Illustration: for loop - hint: vectorized version is possible
        ################################################################################################################
        # What I have been thinking: To calculate the loo_error,                                                       #
        # I have to calculate the error for both children (left, right)                                                #
        # However, I do not understand how is this operation vectorized                                                #
        # Approach: Calculate m, M in a vectorized way, such that I would get len(tj) m's and M's                      #
        ################################################################################################################

        for t in tj:
            #
            Mj_left, mj_right = np.max(np.where(data_unique < t)), np.min(np.where(data_unique > t))

            # Calculate volume of children nodes
            m_left, M_right = m, M
            M_left = copy.deepcopy(M)
            M_left[j] = Mj_left
            m_right = copy.deepcopy(m)
            m_right[j] = mj_right

            vol_left = np.prod(M_left - m_left)
            vol_right = np.prod(M_right - m_right)

            # Compute the error
            loo_error = (n / (N * vol_left)) * (n / N - 2 * ((n - 1) / (N - 1))) + (n / (N * vol_right)) * (n / N - 2 * ((n - 1) / (N - 1)))

            # choose the best threshold that
            if loo_error < e_min:
                e_min = loo_error
                j_min = j
                t_min = t

    # create children
    left = BaseClasses.Node()
    right = BaseClasses.Node()

    # initialize 'left' and 'right' with the data subsets and bounding boxes
    # according to the optimal split found above
    left.data = ...  # store data in left node -- for subsequent splits
    left.box = ...  # store bounding box in left node
    right.data = ...
    right.box = ...

    # turn the current 'node' into a split node
    # (store children and split condition)
    node.left = left
    node.right = right
    node.feature = ...
    node.threshold = ...

    # return the children (to be placed on the stack)
    return left, right


def make_density_leaf_node(node, N):
    """

    :param node: the node to become a leaf
    :param N: the total number of training instances for the current class
    :return:
    """
    # compute and store leaf response
    n = node.data.shape[0]
    v = ...
    node.response = ...


def test():
    tree = BaseClasses.Tree()
    dt = DensityTree()

    digits = load_digits()
    x_training, x_test, y_training, y_test = gen.data_preparation(digits, 0.33, 0)
    prior = 1 / 2
    dt.train(x_training, prior)


if __name__ == '__main__':
    test()
