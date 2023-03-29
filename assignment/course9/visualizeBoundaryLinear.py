import numpy as np
from plotData import plot_data
from matplotlib import pyplot as plt


def visualize_boundary_linear(x, y, model):
    """
    visualize_boundary_linear plots a linear decision boundary learned by the
    SVM
    visualize_boundary_linear(X, y, model) plots a linear decision boundary
    learned by the SVM and overlays the data on it
    :param x:
    :param y:
    :param model:
    :return:
    """
    u = np.linspace(min(x[:, 0]), max(x[:, 0]), 500)
    v = np.linspace(min(x[:, 1]), max(x[:, 1]), 500)
    plot_data(x, y)
    x, y = np.meshgrid(u, v)
    z = model.predict(np.c_[x.flatten(), y.flatten()])
    z = z.reshape(x.shape)
    plt.contour(x, y, z, 1, colors="b")
