# Machine Learning Online Class
#   Exercise 1: Linear regression with multiple variables
#
#   Instructions
#   ------------
#
#   This file contains code that helps you get started on the
#   linear regression exercise.
#
#   You will need to complete the following functions in this
#   exercise:
#
#      warmUpExercise.m
#      plotData.m
#      gradientDescent.m
#      computeCost.m
#      gradientDescentMulti.m
#      computeCostMulti.m
#      featureNormalize.m
#      normalEqn.m
#
#   For this part of the exercise, you will need to change some
#   parts of the code below for various experiments (e.g., changing
#   learning rates).
#
#
#   Initialization

import numpy as np
from matplotlib import pyplot as plt

from featureNormalize import feature_normalize
from gradientDescentMulti import gradient_descent_multi
from normalEqn import normal_eqn


# ================ Part 1: Feature Normalization ================
def main():
    print("Loading data ...")

    # Load Data
    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m = len(y)

    # Print out some data points
    print("First 10 examples from the dataset: \n")
    print(" x = {}, y = {}".format(X[0:10, :], y[0:10]))

    input("Program paused. Press enter to continue. \n")

    # Scale features and set them to zero mean
    print("Normalizing Features ...")

    X, mu, sigma = feature_normalize(X)

    # Add intercept term to X
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # ================ Part 2: Gradient Descent ================
    # ====================== YOUR CODE HERE ======================
    #  Instructions: We have provided you with the following starter
    #                code that runs gradient descent with a particular
    #                learning rate (alpha).
    #
    #                Your task is to first make sure that your functions -
    #                computeCost and gradientDescent already work with
    #                this starter code and support multiple variables.
    #
    #                After that, try running gradient descent with
    #                different values of alpha and see which one gives
    #                you the best result.
    #
    #                Finally, you should complete the code at the end
    #                to predict the price of a 1650 sq-ft, 3 br house.
    #
    #  Hint: By using the 'hold on' command, you can plot multiple
    #        graphs on the same figure.
    #
    #  Hint: At prediction, make sure you do the same feature normalization.

    print("Running gradient descent ...")

    # Choose some alpha value
    alpha = 0.01
    num_iters = 1500

    # Init Theta and Run Gradient Descent
    theta = np.zeros(3)
    [theta, J_history] = gradient_descent_multi(X, y, theta, alpha, num_iters)

    # Plot the convergence graph
    plt.plot(range(num_iters), J_history, '-b', linewidth=2)
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost J')
    plt.show()

    # Display gradient descent's result
    print("Theta computed from gradient descent: ")
    print(theta)

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================
    # Recall that the first column of X is all-ones. Thus, it does
    # not need to be normalized.
    x_t = np.array([1650, 3])
    x_t[0] = (x_t[0] - mu[0]) / sigma[0]
    x_t[1] = (x_t[1] - mu[1]) / sigma[1]
    x_t = np.concatenate((np.ones(1), x_t))
    price = (x_t * theta).sum()  # You should change this

    # ============================================================

    print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent): ${}'.format(price))

    input("Program paused. Press enter to continue.")

    # ================ Part 3: Normal Equations ================

    print("Solving with normal equations...")

    # ====================== YOUR CODE HERE ======================
    # Instructions: The following code computes the closed form
    #               solution for linear regression using the normal
    #               equations. You should complete the code in
    #               normalEqn.m
    #
    #               After doing so, you should complete this code
    #               to predict the price of a 1650 sq-ft, 3 br house.

    data = np.loadtxt('ex1data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]
    m = len(y)

    # Add intercept term to X
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Calculate the parameters from the normal equation
    theta = normal_eqn(X, y)

    # Display normal equation's result
    print("Theta computed from the normal equations: \n{}".format(theta))

    # Estimate the price of a 1650 sq-ft, 3 br house
    # ====================== YOUR CODE HERE ======================
    x_t = np.array([1650, 3])
    x_t = np.concatenate((np.ones(1), x_t))
    price = (x_t * theta).sum()

    # ============================================================
    print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations): ${}'.format(price))


if __name__ == '__main__':
    main()
