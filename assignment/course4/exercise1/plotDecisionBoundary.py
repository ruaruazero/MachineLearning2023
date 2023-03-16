from matplotlib import pyplot as plt
import numpy as np


def plot_decision_boundary(theta, x, y):
    """
    plot_decision_boundary Plots the data points X and y into a new figure with
    the decision boundary defined by theta
    plot_decision_boundary(theta, X,y) plots the data points with + for the
    positive examples and o for the negative examples. X is assumed to be
    an either
    1) Mx3 matrix, where the first column is an all-ones column for the
       intercept.
    2) MxN, N>3 matrix, where the first column is all-ones
    :return:
    """

    # Plot data points
    fig = plt.figure(dpi=200)
    plt.scatter(x[np.where(y == 0), 1], x[np.where(y == 0), 2], marker="o", color='y', label="Not admitted")
    plt.scatter(x[np.where(y == 1), 1], x[np.where(y == 1), 2], marker="+", color='k', label="Admitted")

    # Define the range of x values to plot the decision boundary
    x_range = np.array([np.min(x[:, 1]) - 2, np.max(x[:, 1]) + 2])

    # Calculate the y values of the decision boundary
    y_range = (-1.0 / theta[2]) * (theta[1] * x_range + theta[0])

    # Plot the decision boundary
    plt.plot([x_range[0], y_range[0]], [x_range[1], y_range[1]], color='r', linewidth=2, label="Decision Boundary")

    # Set the x and y limits of the plot
    plt.xlim(30, 100)
    plt.ylim(30, 100)

    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')

    # Add legend and show the plot
    plt.legend(('Not Admitted', 'Admitted', 'Decision Boundary'), loc='upper right')
    plt.show()
