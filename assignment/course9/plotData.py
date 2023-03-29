from matplotlib import pyplot as plt
import numpy as np


def plot_data(x, y):
    """
    plot_data Plots the data points x and y into a new figure
        plot_data(x,y) plots the data points and gives the figure axes labels of
        population and profit.
    :param x:
    :param y:
    :return:
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Plot the training data into a figure using the
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.
    #
    # Hint: You can use the 'rx' option with plot to have the markers
    #       appear as red crosses. Furthermore, you can make the
    #       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

    plt.figure()
    plt.scatter(x[np.where(y == 0)[0], 0], x[np.where(y == 0)[0], 1], marker="o", color='y', linewidths=0.5)
    plt.scatter(x[np.where(y == 1)[0], 0], x[np.where(y == 1)[0], 1], marker="+", color='k', linewidths=0.5)
    plt.legend(loc="upper right", labels=["y = 0", "y = 1"])


    # ============================================================
