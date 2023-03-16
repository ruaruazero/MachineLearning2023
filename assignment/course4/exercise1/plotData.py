from matplotlib import pyplot as plt
import numpy as np


def plot_data(x, y):
    plt.scatter(x[np.where(y == 0), 0], x[np.where(y == 0), 1], marker="o", color='y', label="Not admitted")
    plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], marker="+", color='k', label="Admitted")
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.legend(loc="upper right")
    plt.show()
