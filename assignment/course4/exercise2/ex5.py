# Machine Learning Online Class
# Exercise 5 | Regularized Linear Regression and Bias-Variance
#
# Instructions
#  ------------
#
# This file contains code that helps you get started on the
# exercise. You will need to complete the following functions:
#
#    linearRegCostFunction.m
#    learningCurve.m
#    validationCurve.m
#
# For this exercise, you will not need to change any code in this file,
# or any other files other than those mentioned above.

# Initialization
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
from linearRegCostFunction import linear_reg_cost_function
from trainLinearReg import train_linear_reg
from learningCurve import learning_curve


# =========== Part 1: Loading and Visualizing Data =============
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment and plot
# the data.

def main():
    # Load Training Data
    print("Loading and Visualizing Data ...")

    # Load from
    # You will have X, y, Xval, yval, Xtest, ytest in your environment
    data = scio.loadmat("./ex5data1.mat")
    X = np.array(data["X"])
    y = np.array(data["y"]).reshape(X.shape[0])
    Xtest = np.array(data["Xtest"])
    ytest = np.array(data["ytest"]).reshape(Xtest.shape[0])
    Xval = np.array(data["Xval"])
    yval = np.array(data["yval"]).reshape(Xval.shape[0])

    # m = Number of examples
    m = X.shape[0]

    # Plot training data
    plt.scatter(X, y, c="red", linewidths=0.5, marker="x")
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.show()

    input("Program paused. Press enter to continue.")

    # =========== Part 2: Regularized Linear Regression Cost =============
    # You should now implement the cost function for regularized linear
    # regression.

    theta = np.ones(2)
    J, grad = linear_reg_cost_function(theta, np.concatenate((np.ones((m, 1)), X), axis=1), y, 1)

    print("Cost at theta = [1 ; 1]: %.4f \n (this value should be about 303.993192)" % J)

    input("Program paused. Press enter to continue.")

    # =========== Part 3: Regularized Linear Regression Gradient =============
    # You should now implement the gradient for regularized linear
    # regression.

    print("Gradient at theta = [1 ; 1]:  %.4f; %.4f (this value should be about [-15.303016; 598.250744])" % (grad[0],
                                                                                                              grad[1]))
    input("Program paused. Press enter to continue.")

    # =========== Part 4: Train Linear Regression =============
    # Once you have implemented the cost and gradient correctly, the
    # trainLinearReg function will use your cost function to train
    # regularized linear regression.
    #
    # Write Up Note: The data is non-linear, so this will not give a great
    #                fit.

    # Train linear regression with lambda = 0
    lam = 0
    theta = train_linear_reg(np.concatenate((np.ones((m, 1)), X), axis=1), y, lam)

    # Plot fit over the data
    plt.scatter(X, y, c="red", linewidths=0.5, marker="x")
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(X, (np.concatenate((np.ones((m, 1)), X), axis=1) * theta).sum(axis=1), '-')
    plt.show()

    input("Program paused. Press enter to continue.")

    # =========== Part 5: Learning Curve for Linear Regression =============
    # Next, you should implement the learningCurve function.
    #
    # Write Up Note: Since the model is under-fitting the data, we expect to
    #                see a graph with "high bias" -- slide 8 in ML-advice.pdf

    lam = 0
    error_train, error_val = learning_curve(
        np.concatenate((np.ones((m, 1)), X), axis=1),
        y,
        np.concatenate((np.ones((Xval.shape[0], 1)), Xval), axis=1),
        yval,
        lam
    )
    x_range = [i for i in range(m)]
    plt.plot(x_range, error_train, c="blue")
    plt.plot(x_range, error_val, c="red")
    plt.title('Learning curve for linear regression')
    plt.legend(['Train', 'Cross Validation'])
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.show()

    print("# Training Examples\tTrain Error\tCross Validation Error")
    for i in range(m):
        print('  \t%d\t\t%f\t%f\n' % (i, error_train[i], error_val[i]))

    input("Program paused. Press enter to continue.")


if __name__ == "__main__":
    main()