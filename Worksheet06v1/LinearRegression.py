from scipy.sparse.linalg import lsqr
import numpy as np


class LinearRegression:
    def __init__(self, df_mean):
        self.theta = None
        self.df_mean = df_mean

    def train(self, features, labels):
        self.theta = lsqr(features, labels)[0]

    def predict(self, x):
        x_mean = self.df_mean.drop(labels=['percentageReds'])
        y_mean = self.df_mean['percentageReds']
        return y_mean + np.sum(self.theta * (x-x_mean))

