import numpy as np


def sigmoid(z: np.array):
    """
    SIGMOID Compute sigmoid function
    J = SIGMOID(z) computes the sigmoid of z.
    :param z:
    :return:
    """

    # You need to return the following variables correctly
    g = np.zeros(z.shape[0])

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the sigmoid of each value of z (z can be a matrix,
    #               vector or scalar).
    if len(z.shape) > 1:
        z = z.sum(axis=1)
    else:
        z = z.sum()

    g = 1 / (1 + np.exp(-z))

    return g

    # =============================================================
