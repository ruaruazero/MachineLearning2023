# This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exercise:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s

import numpy as np
from matplotlib import pyplot as plt

from warmUpExercise import warm_up_exercise
from plotData import plot_data
from computeCost import compute_cost
from gradientDescent import gradient_descent


def main():
    # ====================== Part 1: Basic Function ==================
    # Complete warmUpExercise.py
    print("Running warmUpExercise ...")
    print("5x5 Identity Matrix: ")

    warm_up_exercise()

    input("Program paused. Press enter to continue. \n")

    # ======================= Part 2: Plotting =======================
    print('Plotting Data ...')
    data = np.loadtxt('ex1data1.txt', delimiter=',')
    X = data[:, 0]
    y = data[:, 1]
    m = len(y)

    # Plot Data
    # Note: You have to complete the code in plotData.py
    plot_data(X, y)

    input('Program paused. Press enter to continue.\n')

    # =================== Part 3: Gradient descent ===================
    print('Running Gradient Descent ...')

    X = np.column_stack((np.ones(m), data[:, 0]))  # Add a column of ones to X
    theta = np.zeros(2)  # initialize fitting parameters

    # Some gradient descent settings
    iterations = 1500
    alpha = 0.01

    # Compute and display initial cost
    compute_cost(X, y, theta)

    # Run gradient descent
    gradient_descent(X, y, theta, alpha, iterations)

    # Print theta to screen
    print('Theta found by gradient descent: {:.4f}, {:.4f}'.format(theta[0], theta[1]))

    # Plot the linear fit
    fig = plt.figure(dpi=150)
    plt.scatter(X[:, 1], y, marker="x", c="red", linewidths=0.5)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.title("Scatter plot of training data")
    plt.plot(X[:, 1], X.dot(theta), '-')
    plt.legend(['Training data', 'Linear regression'])
    plt.show()

    # Predict values for population sizes of 35,000 and 70,000
    predict1 = np.array([1, 3.5]).dot(theta)
    print('For population = 35,000, we predict a profit of {:.2f}\n'.format(predict1 * 10000))
    predict2 = np.array([1, 7]).dot(theta)
    print('For population = 70,000, we predict a profit of {:.2f}\n'.format(predict2 * 10000))

    input('Program paused. Press enter to continue.\n')

    # ============= Part 4: Visualizing J(theta_0, theta_1) =============
    print('Visualizing J(theta_0, theta_1) ...')

    # Grid over which we will calculate J
    theta0_vals = np.linspace(-10, 10, 100)
    theta1_vals = np.linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

    # Fill out J_vals
    for i in range(len(theta0_vals)):
        for j in range(len(theta1_vals)):
            t = np.array([theta0_vals[i], theta1_vals[j]])
            J_vals[i, j] = compute_cost(X, y, t)

    # Surface plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(theta0_vals, theta1_vals, J_vals)
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')
    plt.show()

    # Contour plot
    plt.figure()
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    plt.contour(theta0_vals, theta1_vals, J_vals, levels=np.logspace(-2, 3, 20))
    plt.xlabel(r'$\theta_0$')
    plt.ylabel(r'$\theta_1$')

    plt.plot(theta[0], theta[1], 'rx', markersize=10, linewidth=2)
    plt.show()


if __name__ == '__main__':
    main()
