import Worksheet05.DecisionTree as DecisionTree
import Worksheet05.DecisionForest as DecisionForest
import Worksheet05.GenerativeClassifier as GenerativeClassifier
import Worksheet05.GenerativeClassifierDensityForest as GenerativeClassifierDensityForest

from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier

import numpy as np

import matplotlib.pyplot as plt
from IPython.display import display

import pandas as pd

pd.options.display.float_format = '{:,.2f}'.format


def task2(data, data_subsets, target):
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

    """
    In the confusion matrices above, the diagonals give the accuracies of predictions of digits, the off-diagonal 
    elements are the error cases. You can see that the discriminative classifiers with decision trees always perform 
    better than the generative classifiers with density trees, reflecting the fact that accurate generative modeling is 
    harder than discriminative modeling.
    Decreasing n_min increases the training accuracy of both classifiers, but this does not imply that the test accuracy
    would also increase! Also if we repeat the experiments, the results can change significantly, because the success 
    rate highly depends on the random selection of feature subsets when searching for the optimal split in each node.
    """

    """
    Here, we try to drop the size restriction of the split nodes, i.e. to train all leaves to purity, by setting 
    hyperparameter n_min = 0. As expected, this results in 100% training accuracy. Do not forget, that this doesn't mean
    that the test error decreased as well -- in fact, it will typically increase due to overfitting.
    """

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


def task3(data, data_subsets, target):
    n_min = [20, 10]

    for n in n_min:
        # training generative classifier with DensityForests
        GCF = GenerativeClassifierDensityForest.GenerativeClassifierDensityForest(20)
        GCF.train(data, target, n)

        # traing a discriminative classifier using one instance of DecisionForest
        df = DecisionForest.DecisionForest(20)
        df.train(data, target, n)

        # traing  sklearn's predefined decision forest sklearn.ensemble.RandomForestClassifier
        RFC = RandomForestClassifier(20, min_samples_split=n)
        RFC.fit(data, target)

        # data_subsets = [data[target==i] for i in  range(10)]
        confusion_GC_Forest = np.zeros((10, 10))
        confusion_DC_Forest = np.zeros((10, 10))
        confusion_RFC = np.zeros((10, 10))
        # for each digit subset make prediction and plot confusion matrix
        for i in range(10):
            # predict with generative classifier
            predictions = np.array([GCF.predict(j) for j in data_subsets[i]])
            confusion_GC_Forest[i, :] = np.bincount(predictions, minlength=10) / len(data_subsets[i]) * 100
            # predict with discriminative classifier
            predictions = np.array([np.argmax(df.predict(i)) for i in data_subsets[i]])
            confusion_DC_Forest[i, :] = np.bincount(predictions, minlength=10) / len(data_subsets[i]) * 100

            # predict for sklearn RF classifier
            predictions = RFC.predict(data_subsets[i])
            confusion_RFC[i, :] = np.bincount(predictions, minlength=10) / len(data_subsets[i]) * 100

        print('Confusion Matrix for Generative Classifier using 10 instances of Density Forest')

        print('-----------------------------------------------------------------------------------------')
        display(
            pd.DataFrame(data=confusion_GC_Forest, index=range(10), columns=range(10)).rename_axis('Actual/Predicted',
                                                                                                   axis='columns')
        )
        print('Confusion Matrix for Discriminative Classifier using 1 instance of Decision Forest')

        print('-----------------------------------------------------------------------------------------')
        display(
            pd.DataFrame(data=confusion_DC_Forest, index=range(10), columns=range(10)).rename_axis('Actual/Predicted',
                                                                                                   axis='columns')
        )
        print('Confusion Matrix for sklearn\'s RandomForest')

        print('-----------------------------------------------------------------------------------------')
        display(
            pd.DataFrame(data=confusion_RFC, index=range(10), columns=range(10)).rename_axis('Actual/Predicted',
                                                                                             axis='columns')
        )


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
    task2(data, data_subsets, target)

    task3(data, data_subsets, target)


if __name__ == '__main__':
    main()
