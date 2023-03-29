import numpy as np


def poly_features(x, p):
    """
    poly_features Maps X (1D vector) into the p-th power
    [X_poly] = poly_features(X, p) takes a data matrix X (size m x 1) and
    maps each example into its polynomial features where
    X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    :return:
    """

    # You need to return the following variables correctly.
    x_poly = np.zeros((x.shape[0], p))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Given a vector X, return a matrix X_poly where the p-th
    #               column of X contains the values of X to the p-th power.
    for i in range(p):
        x_poly[:, i] = x.reshape(x.shape[0]) ** (i + 1)

    return x_poly

    # =========================================================================
