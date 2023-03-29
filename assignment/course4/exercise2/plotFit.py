from matplotlib import pyplot as plt
import numpy as np
from polyFeatures import poly_features

def plot_fit(min_x, max_x, mu, sigma, theta, p):
    """
    plot_fit Plots a learned polynomial regression fit over an existing figure.
    Also works with linear regression.
    plot_fit(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    fit with power p and feature normalization (mu, sigma).
    :param min_x:
    :param max_x:
    :param mu:
    :param sigma:
    :param theta:
    :param p:
    :return:
    """
    # Plotting the fit

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05).reshape(-1, 1)

    # Map the X values
    X_poly = poly_features(x, p)
    X_poly = (X_poly - mu) / sigma

    # Add ones
    X_poly = np.hstack((np.ones((x.shape[0], 1)), X_poly))

    # Plot
    plt.plot(x, X_poly @ theta, '--', linewidth=2)

    # Hold off to the current figure



