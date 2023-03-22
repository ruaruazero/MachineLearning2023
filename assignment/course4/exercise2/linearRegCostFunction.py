import numpy as np


def linear_reg_cost_function(theta, x, y, lam):
    """
    linear_reg_cost_function Compute cost and gradient for regularized linear
    regression with multiple variables
        [J, grad] = linear_reg_cost_function(X, y, theta, lambda) computes the
        cost of using theta as the parameter for linear regression to fit the
        data points in X and y. Returns the cost in J and the gradient in grad
    :param x:
    :param y:
    :param theta:
    :param lam:
    :return:
    """

    # Initialize some useful values
    m = len(y)

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost and gradient of regularized linear
    #               regression for a particular choice of theta.
    #
    #               You should set J to the cost and grad to the gradient.
    t_theta = theta.copy()
    t_theta[0] = 0
    h = (x * theta).sum(axis=1)
    J = (((h - y) ** 2).sum() + lam * (theta ** 2).sum()) / (2 * m)
    tmp = (h - y).reshape((m, 1))
    grad = ((tmp * x).sum(axis=0) + lam * t_theta) / m
    return J, grad
