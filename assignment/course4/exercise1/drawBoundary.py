import numpy as np
from matplotlib import pyplot as plt
from mapFeature import map_feature


def draw_boundary(theta, x, y):
    plt.scatter(x[np.where(y == 0), 0], x[np.where(y == 0), 1], marker="o", color='y', label="y = 0")
    plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], marker="+", color='k', label="y = 1")
    delta = 0.05
    x_range = np.arange(-1, 1.5, delta)
    y_range = np.arange(-1, 1.5, delta)
    x_range, y_range = np.meshgrid(x_range, y_range)
    z = np.zeros(x_range.shape)
    for i in range(y_range.shape[0]):
        z[i, :] = (map_feature(x_range[0, :], y_range[i, :]) * theta).sum(axis=1)
    print(z)
    c = plt.contour(x_range, y_range, z, 0, colors="r")
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.legend()
    plt.show()
