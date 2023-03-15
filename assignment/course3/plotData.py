import numpy as np
from matplotlib import pyplot as plt


def plot_data(x: np.array, y: np.array):
    """
    plot_data Plots the data points x and y into a new figure
    plot_data(x,y) plots the data points and gives the figure axes labels of
    population and profit.
    :return:
    """
    # ====================== YOUR CODE HERE ======================
    # instructions是针对Matlab代码, python需要使用matplotlib
    # Instructions: Plot the training data into a figure using the
    #               "figure" and "plot" commands. Set the axes labels using
    #               the "xlabel" and "ylabel" commands. Assume the
    #               population and revenue data have been passed in
    #               as the x and y arguments of this function.

    # Hint: You can use the 'rx' option with plot to have the markers
    #       appear as red crosses. Furthermore, you can make the
    #       markers larger by using plot(..., 'rx', 'MarkerSize', 10);

    fig = plt.figure(dpi=150)
    plt.scatter(x, y, marker="x", c="red", linewidths=0.5)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.title("Scatter plot of training data")
    plt.show()

    # ============================================================
