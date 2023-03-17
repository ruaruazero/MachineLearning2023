import numpy as np

from computeCostMulti import compute_cost_multi


def gradient_descent_multi(x, y, theta, lr, num_iters):
    """
    gradient_descent_multi Performs gradient descent to learn theta
    theta = gradient_descent_multi(x, y, theta, alpha, num_iters) updates theta by
    taking num_iters gradient steps with learning rate alpha
    :return:
    """
    # Initialize some useful values
    m = len(y)
    J_history = np.zeros(num_iters)

    for iter_time in range(num_iters):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Perform a single gradient step on the parameter vector
        #               theta.
        #
        # Hint: While debugging, it can be useful to print out the values
        #       of the cost function (computeCostMulti) and gradient here.

        h = (theta * x).sum(axis=1) - y
        dl_dtheta = ((1 / m) * (h.reshape(h.shape[0], 1) * x)).sum(axis=0)
        # dl_dw = ((h * x[:, 1]).sum()) / m
        # dl_dt = ((h * x[:, 2]).sum()) / m
        # dl_db = (h.sum()) / m
        theta[0] = theta[0] - lr * dl_dtheta[0]
        theta[1] = theta[1] - lr * dl_dtheta[1]
        theta[2] = theta[2] - lr * dl_dtheta[2]

        # ============================================================

        # Save the cost J in every iteration
        J_history[iter_time] = compute_cost_multi(x, y, theta)

    return theta, J_history
