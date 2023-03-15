def compute_cost(x, y, theta):
    """
    compute_cost Compute cost for linear regression
        J = compute_cost(X, y, theta) computes the cost of using theta as the
        parameter for linear regression to fit the data points in X and y
    :return:
    """
    # Initialize some useful values
    m = len(y)  # number of training examples

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.

    h = x * theta
    J += ((h.sum(axis=1) - y)**2).sum() / (2 * m)

    print(J)
    return J
    # =========================================================================
