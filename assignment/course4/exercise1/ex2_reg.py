# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
import copy

# Initialization
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as opt

from plotData import plot_data
from mapFeature import map_feature
from costFunctionReg import cost_function_reg
from drawBoundary import draw_boundary
from predict import predict

np.random.seed(1)


# Load Data
def main_reg():
    data = np.loadtxt('ex2data2.txt', delimiter=',')
    X = data[:, 0:2]
    y = data[:, 2]

    X_init = copy.deepcopy(X)

    plot_data(X, y)

    plt.show()

    # =========== Part 1: Regularized Logistic Regression ============
    # Add Polynomial Features

    # Note that mapFeature also adds a column of ones for us, so the intercept term is handled
    X = map_feature(X[:, 0], X[:, 1])

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])

    # Set regularization parameter lambda to 1
    lambda_ = 1

    # Compute and display initial cost and gradient for regularized logistic regression
    cost, grad = cost_function_reg(initial_theta, X, y, lambda_)
    print(f'Cost at initial theta (zeros): {cost}')

    # ============= Part 2: Regularization and Accuracies =============
    # Optional Exercise:
    # In this part, you will get to try different values of lambda and see how regularization affects the decision boundary.
    # Try the following values of lambda (0, 1, 10, 100).

    # Initialize fitting parameters
    initial_theta = np.zeros(X.shape[1])

    # Set regularization parameter lambda to 1 (you should vary this)
    lambda_ = 1

    # Set Options
    options = {'maxfun': 400}

    # Optimize
    res = opt.minimize(cost_function_reg, initial_theta, args=(X, y, lambda_), jac=True, method='TNC', options=options)
    theta = res.x

    # Plot Boundary
    draw_boundary(theta, X_init, y)

    # Labels and Legend


    # Compute accuracy on our training set
    p = predict(theta, X)
    print(f'Train Accuracy: {np.mean(p == y) * 100}')


if __name__ == '__main__':
    main_reg()
