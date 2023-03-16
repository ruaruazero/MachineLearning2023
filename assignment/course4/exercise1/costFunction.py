import numpy as np

from sigmoid import sigmoid


def cost_function(theta, x, y):
    """
    cost_function Compute cost and gradient for logistic regression
    J = cost_function(theta, X, y) computes the cost of using theta as the
    parameter for logistic regression and the gradient of the cost
    w.r.t. to the parameters.
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
    #
    # Note: grad should have the same dimensions as theta
    z = x * theta
    h_theta = sigmoid(z)
    J = y * np.log(h_theta) + (1 - y) * np.log(1 - h_theta)
    J = J.sum() * (-1 / m)
    dl_dr = ((1 / m) * (h_theta - y) * x[:, 1]).sum()
    dl_dt = ((1 / m) * (h_theta - y) * x[:, 2]).sum()
    dl_dw = ((1 / m) * (h_theta - y) * x[:, 0]).sum()

    grad = np.array([dl_dw, dl_dr, dl_dt])

    return J, grad
