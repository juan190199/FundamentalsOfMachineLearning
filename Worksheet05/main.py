import Worksheet05.DecisionTree as DecisionTree
import Worksheet05.GenerativeClassifier as GenerativeClassifier

from sklearn.datasets import load_digits

import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd

pd.options.display.float_format = '{:,.2f}'.format


def main():
    # Load digits
    digits = load_digits()
    data = digits["data"]
    target = digits["target"]

    # Data subsets for each digit
    data_subsets = [data[target == i] for i in range(10)]

    # Below we will train a GenerativeClassifier as discussed and a discriminative classifier with one
    # DecisionTree with full dataset.
    # And then plot the full training error confusion matrices.

    n_min = [20, 10]

    for n in n_min:
        # Train with full dataset
        # Training generative classifier with DensityTrees
        GC = GenerativeClassifier.GenerativeClassifier()
        GC.train(data, target, n)
        # Training a discriminative classifier using one instance of DecisionTree
        dt = DecisionTree.DecisionTree()
        dt.train(data, target, n)

        # Predict and compute the full training error confusion matrices
        confusion_GC = np.zeros((10, 10))
        confusion_DC = np.zeros((10, 10))
        # For each digit subset
        for i in range(10):
            # Predict for with generative classifier
            predictions = np.array([GC.predict(j) for j in data_subsets[i]])
            confusion_GC[i, :] = np.bincount(predictions, minlength=10) / len(data_subsets[i]) * 100
            # Predict for with discriminative tive classifier
            predictions = np.array([np.argmax(dt.predict(j)) for j in data_subsets[i]])
            confusion_DC[i, :] = np.bincount(predictions, minlength=10) / len(data_subsets[i]) * 100

        print('Confusion Matrix for Generative Classifier using 10 instances of Density Tree')

        print('-------------------------------------------------------------------------------')
        display(
            pd.DataFrame(data=confusion_GC, index=range(10), columns=range(10)).rename_axis('Actual/Predicted',
                                                                                            axis='columns')
        )
        print('Confusion Matrix for Discriminative Classifier using 1 instance of Decision Tree')

        print('-------------------------------------------------------------------------------')
        display(
            pd.DataFrame(data=confusion_DC, index=range(10), columns=range(10)).rename_axis('Actual/Predicted',
                                                                                            axis='columns')
        )

    explanation = """
    In the confusion matrices above, the diagonals give the accuracies of predictions of digits, the off-diagonal 
    elements are the error cases. You can see that the discriminative classifiers with decision trees always perform 
    better than the generative classifiers with density trees, reflecting the fact that accurate generative modeling is 
    harder than discriminative modeling.
    Decreasing n_min increases the training accuracy of both classifiers, but this does not imply that the test accuracy
    would also increase! Also if we repeat the experiments, the results can change significantly, because the success 
    rate highly depends on the random selection of feature subsets when searching for the optimal split in each node.
    """

    print(explanation.ljust(40, '-'))

    # Discriminative classifier using one instance of DecisionTree
    dt = DecisionTree.DecisionTree()
    dt.train(data, target, n_min=0)

    confusion_DC = np.zeros((10, 10))
    # for each digit subset
    for i in range(10):
        # predict for generative classifier
        predictions = np.array([np.argmax(dt.predict(j)) for j in data_subsets[i]])
        confusion_DC[i, :] = np.bincount(predictions, minlength=10) / len(data_subsets[i]) * 100

    print('Confusion Matrix for Discriminative Classifier using 10 instances of Decision Tree')

    print('----------------------------------------------------------------------------------')
    display(
        pd.DataFrame(data=confusion_DC, index=range(10), columns=range(10)).rename_axis('Actual/Predicted',
                                                                                        axis='columns')
    )


if __name__ == '__main__':
    main()
