import numpy as np
from scipy import optimize as opt
from linearRegCostFunction import linear_reg_cost_function


def train_linear_reg(x, y, lam):
    """
    train_linear_reg Trains linear regression given a dataset (X, y) and a
    regularization parameter lambda
        [theta] = train_linear_reg(X, y, lambda) trains linear regression using
        the dataset (X, y) and regularization parameter lambda. Returns the
        trained parameters theta.
    :param x:
    :param y:
    :param lam:
    :return:
    """
    # Initialize Theta
    initial_theta = np.zeros(x.shape[1])

    # Create "short hands" for the cost function to be minimized

    options = {'maxfun': 200}
    res = opt.minimize(linear_reg_cost_function, initial_theta, (x, y, lam), method='TNC', jac=True, options=options)

    theta = res.x

    return theta
