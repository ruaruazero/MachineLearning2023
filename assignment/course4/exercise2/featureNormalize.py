import numpy as np


def feature_normalize(x: np.array):
    """
    feature_normalize Normalizes the features in X
       feature_normalize(X) returns a normalized version of X where
       the mean value of each feature is 0 and the standard deviation
       is 1. This is often a good preprocessing step to take when
       working with learning algorithms.
    :return:
    """

    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    x_norm = (x - mu) / sigma

    return x_norm, mu, sigma
