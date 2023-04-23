# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
#
# Instructions
# ------------
#
# This file contains code that helps you get started on the
# linear exercise. You will need to complete the following functions
# in this exercise:
#
#    lrCostFunction.m (logistic regression cost function)
#    oneVsAll.m
#    predictOneVsAll.m
#    predict.m
#
# For this exercise, you will not need to change any code in this file,
# or any other files other than those mentioned above.
from scipy.io import loadmat
import numpy as np
from displayData import display_data


def main():

    # Set up the parameters you will use for this part of the exercise
    input_layer_size = 400
    num_labels = 10

    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset.
    # You will be working with a dataset that contains handwritten digits.

    print('Loading and Visualizing Data ...')

    data = loadmat('ex3data1.mat')
    X = data['X']
    y = data['y']

    m = X.shape[0]

    # Randomly select 100 data points to display
    rand_indices = np.random.randint(0, m, 100, dtype=int)
    sel = X[rand_indices, :]

    display_data(sel)

    input('Program paused. Press enter to continue.')

    # ============ Part 2: Vectorize Logistic Regression ============
    # In this part of the exercise, you will reuse your logistic regression
    # code from the last exercise. You task here is to make sure that your
    # regularized logistic regression implementation is vectorized. After
    # that, you will implement one-vs-all classification for the handwritten
    # digit dataset.

    print('\nTraining One-vs-All Logistic Regression...')

    lambda_ = 0.1


if __name__ == '__main__':
    main()
