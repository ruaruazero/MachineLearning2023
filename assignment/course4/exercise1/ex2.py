# Machine Learning Online Class - Exercise 2: Logistic Regression
#
# Instructions
# ------------
#
# This file contains code that helps you get started on the logistic
# regression exercise. You will need to complete the following functions
# in this exercise:
#
#    sigmoid.m
#    costFunction.m
#    predict.m
#    costFunctionReg.m
#
# For this exercise, you will not need to change any code in this file,
# or any other files other than those mentioned above.

# Initialization
import numpy as np
from scipy import optimize as opt

from plotData import plot_data
from costFunction import cost_function
from plotDecisionBoundary import plot_decision_boundary
from sigmoid import sigmoid
from predict import predict


# Load Data
# The first two columns contains the exam scores and the third column
# contains the label.
def main():
    data = np.loadtxt('ex2data1.txt', delimiter=",")
    X = data[:, 0:2]
    y = data[:, 2]

    # ==================== Part 1: Plotting ====================
    # We start the exercise by first plotting the data to understand the
    # problem we are working with.

    print("Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.")

    plot_data(X, y)

    input("Program paused. Press enter to continue. \n")

    # ============ Part 2: Compute Cost and Gradient ============
    # In this part of the exercise, you will implement the cost and gradient
    # for logistic regression. You need to complete the code in
    # costFunction.py

    # Set up the data matrix appropriately, and add ones for the intercept term
    [m, n] = X.shape

    # Add intercept term to x and X_test
    X = np.concatenate((np.ones((m, 1)), X), axis=1)

    # Initialize fitting parameters
    initial_theta = np.zeros(n + 1)

    # Compute and display initial cost and gradient
    [cost, grad] = cost_function(initial_theta, X, y)

    print("Cost at initial theta (zeros): ", cost)
    print("Gradient at initial theta (zeros): ", grad)

    input("Program paused. Press enter to continue.")

    # ============= Part 3: Optimizing using fminunc  =============
    # In this exercise, you will use a built-in function (fminunc) to find the
    # optimal parameter's theta.

    # Set options for fminunc
    options = {'maxfun': 400}

    # Run fminunc to obtain the optimal theta
    # This function will return theta and the cost
    res = opt.minimize(cost_function, initial_theta, (X, y), method='TNC', jac=True, options=options)
    theta = res.x

    # Print theta to screen
    print('Cost at theta found by fminunc: {:.2f}'.format(res.fun))
    print('theta:')
    print(theta)

    # Plot Boundary
    plot_decision_boundary(theta, X, y)

    input("Program paused. Press enter to continue.")

    # ============== Part 4: Predict and Accuracies ==============
    # After learning the parameters, you'll like to use it to predict the outcomes
    # on unseen data. In this part, you will use the logistic regression model
    # to predict the probability that a student with score 45 on exam 1 and
    # score 85 on exam 2 will be admitted.
    #
    # Furthermore, you will compute the training and test set accuracies of
    # our model.
    #
    # Your task is to complete the code in predict.m
    #
    # Predict probability for a student with score 45 on exam 1
    # and score 85 on exam 2

    print((np.array([1, 45, 85]) * theta))
    prob = sigmoid(np.array([1, 45, 85]) * theta)
    print("For a student with scores 45 and 85, we predict an admission probability of: ", prob)

    p = predict(theta, X)

    print("Train Accuracy: ", np.mean(p) * 100)
    input("Program paused. Press enter to continue.")


if __name__ == '__main__':
    main()
