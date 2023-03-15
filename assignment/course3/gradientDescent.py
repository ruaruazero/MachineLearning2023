import numpy as np

from computeCost import compute_cost


def gradient_descent(x, y, theta, lr, num_iters):
    """
    gradient_descent Performs gradient descent to learn theta
        theta = gradient_descent(X, y, theta, alpha, num_iters) updates theta by
        taking num_iters gradient steps with learning rate alpha
    :param x:
    :param y:
    :param theta:
    :param lr:
    :param num_iters:
    :return:
    """

    # Initialize some useful values
    m = len(y)  # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter_time in range(num_iters):
        # ====================== YOUR CODE HERE ======================
        #      Instructions: Perform a single gradient step on the parameter vector
        #                    theta.
        #
        #      Hint: While debugging, it can be useful to print out the values
        #            of the cost function (computeCost) and gradient here.

        # get wx_i + b - y_i
        h = ((x * theta).sum(axis=1) - y)

        # get gradient
        d_w = ((h * x[:, 1]).sum()) / m
        d_b = h.sum() / m

        # update params
        theta[0] = theta[0] - (d_b * lr)
        theta[1] = theta[1] - (d_w * lr)

        # ============================================================

        J_history[iter_time] = compute_cost(x, y, theta)
