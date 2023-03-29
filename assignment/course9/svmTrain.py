import numpy as np


def svm_train(x, y, c, kernel, tol, max_passes):
    """
    svm_train Trains an SVM classifier using a simplified version of the SMO
    algorithm.
       [model] = svm_train(X, Y, C, kernelFunction, tol, max_passes) trains an
       SVM classifier and returns trained model. X is the matrix of training
       examples.  Each row is a training example, and the jth column holds the
       jth feature.  Y is a column matrix containing 1 for positive examples
       and 0 for negative examples.  C is the standard SVM regularization
       parameter.  tol is a tolerance value used for determining equality of
       floating point numbers. max_passes controls the number of iterations
       over the dataset (without changes to alpha) before the algorithm quits.

    Note: This is a simplified version of the SMO algorithm for training
           SVMs. In practice, if you want to train an SVM classifier, we
           recommend using an optimized package such as:

               LIBSVM   (http://www.csie.ntu.edu.tw/~cjlin/libsvm/)
               SVMLight (http://svmlight.joachims.org/)
    :param x:
    :param y:
    :param c:
    :param kernel:
    :param tol:
    :param max_passes:
    :return:
    """
    model = {}
    if 'tol' not in locals() or tol is None:
        tol = 1e-3

    if 'max_passes' not in locals() or max_passes is None:
        max_passes = 5

    m = x.shape[0]
    n = x.shape[1]

    # Map 0 to -1
    y = y.copy()
    y[np.where(y == 0)[0]] = -1

    # Variables
    alphas = np.zeros(m)
    b = 0
    E = np.zeros(m)
    passes = 0
    eta = 0
    L = 0
    H = 0

    # Pre-compute the Kernel Matrix since our dataset is small
    # (in practice, optimized SVM packages that handle large datasets
    #  gracefully will _not_ do this)
    #
    # We have implemented optimized vectorized version of the Kernels here so
    # that the svm training will run faster.
    if kernel.__name__ == "linear_kernel":
        K = np.dot(x, x.T)
    elif 'gaussian_kernel' in kernel.__name__:
        x2 = np.sum(x ** 2, axis=1)
        K = x2[:, np.newaxis] + x2[np.newaxis, :] - 2 * np.dot(x, x.T)
        K = kernel(1, 0)(K)
    else:
        K = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                K[i, j] = kernel(x[i, :], x[j, :])
                K[j, i] = K[i, j]

    # Train
    print("Training ...")
    dots = 12
    while passes < max_passes:

        num_changed_alphas = 0

        for i in range(m):

            # Calculate Ei = f(x(i)) - y(i) using (2).
            E[i] = b + np.sum(alphas * y * K[:, i]) - y[i]

            if (y[i] * E[i] < -tol and alphas[i] < c) or (y[i] * E[i] > tol and alphas[i] > 0):

                # In practice, there are many heuristics one can use to select
                # the i and j. In this simplified code, we select them randomly.
                j = np.random.randint(m)
                while j == i:
                    j = np.random.randint(m)

                # Calculate Ej = f(x(j)) - y(j) using (2).
                E[j] = b + np.sum(alphas * y * K[:, j]) - y[j]

                # Save old alphas
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                # Compute L and H by (10) or (11).
                if y[i] == y[j]:
                    L = max(0, alphas[j] + alphas[i] - c)
                    H = min(c, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(c, c + alphas[j] - alphas[i])
                if L == H:
                    continue

                # Compute eta by (14).
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                # Compute and clip new value for alpha j using (12) and (15).
                alphas[j] = alphas[j] - (y[j] * (E[i] - E[j])) / eta

                # Clip
                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])

                # Check if change in alpha is significant
                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue

                # Determine value for alpha i using (16).
                alphas[i] = alphas[i] + y[i] * y[j] * (alpha_j_old - alphas[j])
                # Compute b1 and b2 using (17) and (18) respectively.
                b1 = b - E[i] - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - E[j] - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                # Compute b by (19).
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2

                num_changed_alphas = num_changed_alphas + 1

        if num_changed_alphas == 0:
            passes = passes + 1
        else:
            passes = 0

        print('.', end='')
        dots = dots + 1
        if dots > 78:
            dots = 0
            print("\n", end='')

    print("Done! \n\n")

    # Save the model
    idx = alphas > 0
    model['X'] = x[idx, :]
    model['y'] = y[idx]
    model['kernelFunction'] = kernel
    model['b'] = b
    model['alphas'] = alphas[idx]
    model['w'] = np.dot(x.T, alphas * y)

    print(model)

    return model
