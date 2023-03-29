import numpy as np


def linear_kernel(x1, x2):
    """
    linear_kernel returns a linear kernel between x1 and x2
        sim = linearKernel(x1, x2) returns a linear kernel between x1 and x2
        and returns the value in sim
    :param x1:
    :param x2:
    :return:
    """
    x1 = x1.reshape(-1, 1)
    x2 = x2.reshape(-1, 1)

    # Compute the kernel
    sim = np.dot(x1.T, x2)

    return sim

