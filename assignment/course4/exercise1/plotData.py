from matplotlib import pyplot as plt
import numpy as np
import sys


def plot_data(x, y):
    plt.scatter(x[np.where(y == 0), 0], x[np.where(y == 0), 1], marker="o", color='y')
    plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], marker="+", color='k')
    if sys._getframe(1).f_code.co_name == "main":
        plt.legend(loc="upper right", labels=["Not Admitted", "Admitted"])
        plt.xlabel('Exam 1 score')
        plt.ylabel('Exam 2 score')
    else:
        plt.legend(loc="upper right", labels=["y = 0", "y = 1"])
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')
    plt.show()
