import numpy as np


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-((x1 - x2) ** 2).sum() / (2 * sigma ** 2))
