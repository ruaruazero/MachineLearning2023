import numpy as np


def normal_eqn(x, y):
    """
    normal_eqn Computes the closed-form solution to linear regression
    normal_eqn(X,y) computes the closed-form solution to linear
    regression using the normal equations.
    :return:
    """

    # ====================== YOUR CODE HERE ======================
    # Instructions: Complete the code to compute the closed form solution
    #               to linear regression and put the result in theta.

    # ---------------------- Sample Solution ----------------------
    t = np.matrix(x.T @ x)
    t = t.I @ x.T @ y
    theta = np.array([t[0, 0], t[0, 1], t[0, 2]])
    # -------------------------------------------------------------
    return theta
    # ============================================================
