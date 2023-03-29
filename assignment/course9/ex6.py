# Machine Learning Online Class
# Exercise 6 | Support Vector Machines
#
# Instructions
# ------------
#
# This file contains code that helps you get started on the
# exercise. You will need to complete the following functions:
#
#    gaussianKernel.m
#    dataset3Params.m
#    processEmail.m
#    emailFeatures.m
#
# For this exercise, you will not need to change any code in this file,
# or any other files other than those mentioned above.

# Initialization
import numpy as np
import scipy.io as scio
from matplotlib import pyplot as plt
from plotData import plot_data
from svmTrain import svm_train
from linearKernel import linear_kernel
from visualizeBoundaryLinear import visualize_boundary_linear
from sklearn.svm import SVC
from gaussianKernel import gaussian_kernel


def main():
    # =============== Part 1: Loading and Visualizing Data ================
    # We start the exercise by first loading and visualizing the dataset.
    # The following code will load the dataset into your environment and plot
    # the data.

    print("Loading and Visualizing Data ...")

    # Load from ex6data1
    # You will have X, y in your environment
    data = scio.loadmat("./ex6data1.mat")
    X = np.array(data["X"])
    y = np.array(data["y"])

    # Plot training data
    plot_data(X, y)
    plt.show()

    input("Program paused. Press enter to continue.")

    # ==================== Part 2: Training Linear SVM ====================
    # The following code will train a linear SVM on the dataset and plot the
    # decision boundary learned.

    print("Training Linear SVM ...")

    # You should try to change the C value below and see how the decision
    # boundary varies (e.g., try C = 1000)
    C = 1
    model = SVC(C=C, kernel="linear")
    model.fit(X, y.flatten())
    visualize_boundary_linear(X, y, model)
    plt.show()

    input("Program paused. Press enter to continue.")

    # =============== Part 3: Implementing Gaussian Kernel ===============
    # You will now implement the Gaussian kernel to use
    # with the SVM. You should complete the code in gaussianKernel.m
    print("Evaluating the Gaussian Kernel ...")

    x1 = np.array([1, 2, 1])
    x2 = np.array([0, 4, -1])
    sigma = 2
    sim = gaussian_kernel(x1, x2, sigma)

    print(
        "Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :"
        f"\n\t{sim}\n(this value should be about 0.324652)\n")

    input("Program paused. Press enter to continue.")

    # =============== Part 4: Visualizing Dataset 2 ================
    # The following code will load the next dataset into your environment and
    # plot the data.

    print("Loading and Visualizing Data ...")

    # Load from ex6data2:
    # You will have X, y in your environment
    data = scio.loadmat("./ex6data2.mat")
    X = np.array(data["X"])
    y = np.array(data["y"])

    plot_data(X, y)
    plt.show()

    input("Program paused. Press enter to continue.")

    # ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
    # After you have implemented the kernel, we can now use it to train the
    # SVM classifier.

    print("Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...")

    C = 1
    sigma = 0.1

    model = SVC(C=C, gamma=np.power(0.1, -2)/2, kernel='rbf')
    model.fit(X, y.flatten())
    visualize_boundary_linear(X, y, model)
    plt.show()

    input("Program paused. Press enter to continue.")

    # =============== Part 6: Visualizing Dataset 3 ================
    # The following code will load the next dataset into your environment and
    # plot the data.

    print("Loading and Visualizing Data ...")

    data = scio.loadmat("./ex6data3.mat")
    X = np.array(data["X"])
    y = np.array(data["y"])
    Xval = np.array(data["Xval"])
    yval = np.array(data["yval"])

    # Plot training data
    plot_data(X, y)
    plt.show()
    input("Program paused. Press enter to continue.")


    # ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
    #
    # This is a different dataset that you can use to experiment with. Try
    # different values of C and sigma here.






if __name__ == "__main__":
    main()
