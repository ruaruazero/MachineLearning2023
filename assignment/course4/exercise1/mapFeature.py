import numpy as np


def map_feature(x1, x2):
    """
    map_feature Feature mapping function to polynomial features

    map_feature(X1, X2) maps the two input features
    to quadratic features used in the regularization exercise.

    Returns a new feature array with more features, comprising of
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..

    Inputs X1, X2 must be the same size
    :param x1:
    :param x2:
    :return:
    """
    degree = 6
    out = np.ones((x1.shape[0], 1))
    for i in range(1, degree + 1):
        for j in range(i + 1):
            out = np.column_stack((out, ((x1 ** (i - j)) * (x2 ** j))))

    return out
