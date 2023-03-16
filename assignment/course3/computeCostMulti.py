def compute_cost_multi(x, y, theta):
    """
    compute_cost_multi Compute cost for linear regression with multiple variables
    J = compute_cost_multi(X, y, theta) computes the cost of using theta as the
    parameter for linear regression to fit the data points in X and y
    :return:
    """
    # Initialize some useful values
    m = len(y)

    # You need to return the following variables correctly
    J = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost of a particular choice of theta
    #               You should set J to the cost.
    J = (((theta * x).sum(axis=1) - y) ** 2).sum() / (2 * m)

    return J
