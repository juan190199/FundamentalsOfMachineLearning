import numpy as np


class Node:
    pass


class Tree:
    def __init__(self):
        self.root = Node()

    def find_leaf(self, x):
        node = self.root
        while hasattr(node, "feature"):
            j = node.feature
            if x[j] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node


class RegressionTree(Tree):
    def __init__(self, df_mean):
        super(RegressionTree, self).__init__()
        self.df_mean = df_mean

    def train(self, data, labels, n_min=500):
        """

        :param data: feature matrix
        :param labels: ground-truth responses
        :param n_min: termination criterion
        :return:
        """
        N, D = data.shape
        D_try = np.max([int(np.sqrt(D)) - 2, 0])  # how many features to consider for each split decision

        # initialize the root node
        self.root.data = data
        self.root.labels = labels

        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0]  # number of instances in present node
            if (n >= n_min):
                # randomly choose D_try-2 features
                feature_indices = np.random.choice(D, D_try, replace=False)
                feature_indices = np.append(feature_indices, [0, 1, 8])
                # split the node into two
                left, right = make_regression_split_node(node, feature_indices)
                # put the two nodes on the stack
                stack.append(left)
                stack.append(right)
            else:
                make_regression_leaf_node(node, self.df_mean)

    def predict(self, x):
        leaf = self.find_leaf(x)
        return leaf.response


def make_regression_split_node(node, feature_indices):
    """

    :param node: node to be split
    :param feature_indices: ndarray: (D_try,). Contains feature indices to be considered in the present plit
    :return:
    """
    n, D = node.data.shape

    # find best feature j (among 'feature_indices') and best threshold t for the split
    # (mainly copied from "density tree")
    e_min = float("inf")
    j_min, t_min = None, None

    for j in feature_indices:
        data_unique = np.sort(np.unique(node.data[:, j]))
        tj = (data_unique[1:] + data_unique[:-1]) / 2.0

        for t in tj:
            data_left = node.data[:, j].copy()
            labels_left = node.labels[data_left <= t].copy()
            data_left = data_left[data_left <= t]

            data_right = node.data[:, j].copy()
            labels_right = node.labels[data_right > t].copy()
            data_right = data_right[data_right > t]

            # compute mean label value on the left and right
            mean_left = np.mean(labels_left)
            mean_right = np.mean(labels_right)

            # compute sum of squared deviation from mean label
            measure_left = np.sum((labels_left - mean_left) ** 2)
            measure_right = np.sum((labels_right - mean_right) ** 2)

            # Compute decision rule
            measure = measure_left + measure_right

            # choose the best threshold that minimizes gini
            if measure < e_min:
                e_min = measure
                j_min = j
                t_min = t

    # create children
    left = Node()
    right = Node()

    X = node.data[:, j_min]

    # initialize 'left' and 'right' with the data subsets and labels
    # according to the optimal split found above
    left.data = node.data[X <= t_min]  # data in left node
    left.labels = node.labels[X <= t_min]  # corresponding labels
    right.data = node.data[X > t_min]
    right.labels = node.labels[X > t_min]

    # turn the current 'node' into a split node
    # (store children and split condition)
    node.left = left
    node.right = right
    node.feature = j_min
    node.threshold = t_min

    # return the children (to be placed on the stack)
    return left, right


def make_regression_leaf_node(node, df_mean):
    """
    :param node: node to become a leaf
    :return:
    """
    # compute and store leaf response
    node.response = np.mean(node.labels) + df_mean["percentageReds"]
