import numpy as np

import Worksheet05.BaseClasses as BaseClasses


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
        # put root in stack
        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0]  # number of instances in present node
            if n >= n_min:
                # Call 'make_density_split_node()' with 'D_try' randomly selected
                # indices from 'valid_features'. This turns 'node' into a split node
                # and returns the two children, which must be placed on the 'stack'.

                perm = np.random.permutation(len(valid_features))
                left, right = make_density_split_node(node, N, valid_features[perm][:D_try])
                stack.append(left)
                stack.append(right)
            else:
                # Call 'make_density_leaf_node()' to turn 'node' into a leaf node.
                make_density_leaf_node(node, N)

    def predict(self, x):
        # return p(x | y) * p(y) if x is within the tree's bounding box and return 0 otherwise
        m, M = self.box
        if np.any(x < m) or np.any(x > M):
            return 0.0
        else:
            return self.prior * self.find_leaf(x).response


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
    v = np.prod(M - m)

    if v <= 0:
        raise RuntimeError("Zero volume (should not happen)")

    # find best feature j (among 'feature_indices') and best threshold t for the split
    e_min = float("inf")
    j_min, t_min = 0, 0

    for j in feature_indices:
        # For each feature considered, remove duplicate feature values
        # Duplicate feature values have to be removed since there is no possible threshold
        # in between features with the same values
        # (The midpoint of two features with the same value would be the same feature value)

        # Remove duplicate features
        dj = np.sort(np.unique(node.data[:, j]))

        # Compute candidate thresholds
        tj = (dj[1:] + dj[:-1]) / 2

        # Each candidate threshold we need to compute leave-one-out error (looErr) of the resulting children node
        for t in tj:
            # Compute the number of instances in left and right node
            nl = np.sum(node.data[:, j] <= t)
            nr = n - nl

            # Compute volumes of left and right nodes
            vl = t / (M[j] - m[j])  # vl = v * t / (M[j] - m[j])
            vr = 1.0 - vl  # vr = v - vl

            """
            Note: actual volumes would be the commented lines which differ by the constant factor v. 
            A more efficient computation is being used because it will not make any 
            difference in comparing the looErr's for different thresholds.
            """

            if vl == 0 or vr == 0:
                continue

            # Compute looErr's
            el = nl / N / vl * (nl / N - 2.0 * (nl - 1) / (N - 1))
            er = nr / N / vr * (nr / N - 2.0 * (nr - 1) / (N - 1))

            # Choose the best threshold that minimizes sum of looErr
            loo_error = el + er
            if loo_error < e_min:
                e_min = loo_error
                j_min = j
                t_min = t

    # create children
    left = BaseClasses.Node()
    right = BaseClasses.Node()

    # initialize 'left' and 'right' with the data subsets and bounding boxes
    # according to the optimal split found above
    left.data = node.data[node.data[:, j_min] <= t_min, :]
    left.box = m.copy(), M.copy()
    left.box[1][j_min] = t_min
    right.data = node.data[node.data[:, j_min] > t_min, :]
    right.box = m.copy(), M.copy()
    right.box[0][j_min] = t_min

    # Turn the current 'node' into a split node (store children and split condition)
    node.left = left
    node.right = right
    node.feature = j_min
    node.threshold = t_min

    # Return the children (to be placed on the stack)
    return left, right


def make_density_leaf_node(node, N):
    """
    :param node: the node to become a leaf
    :param N: the total number of training instances for the current class
    :return:
    """
    # compute and store leaf response
    n = node.data.shape[0]
    m, M = node.box
    v = np.prod(M - m)
    node.response = n / N / v
