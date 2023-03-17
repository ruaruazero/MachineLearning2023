import numpy as np
from sigmoid import sigmoid


def cost_function_reg(theta, x, y, lam):
    """
    cost_function_reg Compute cost and gradient for logistic regression with regularization
    J = cost_function_reg(theta, X, y, lambda) computes the cost of using
    theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    :param theta:
    :param x:
    :param y:
    :param lam:
    :return:
    """

    # Initialize some useful values
    m = len(y)

    # You need to return the following variables correctly
    J = 0
    grad = np.zeros(theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta.
    #               You should set J to the cost.
    #               Compute the partial derivatives and set grad to the partial
    #               derivatives of the cost w.r.t. each parameter in theta
    h = sigmoid(theta * x)
    J = (1 / m) * (-y * np.log(h) - (1 - y) * np.log(1 - h)).sum() + (lam / 2 / m) * (theta[1:] ** 2).sum()
    grad_reg = (lam / m) * np.concatenate((np.zeros(1), theta[1:]))
    grad = (1 / m) * ((h - y).reshape(x.shape[0], 1) * x).sum(axis=0) + grad_reg

    return J, grad
    # =============================================================
