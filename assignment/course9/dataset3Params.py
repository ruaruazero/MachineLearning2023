import numpy as np
from sklearn.svm import SVC


def dataset_3_params(X, y, Xval, yval):
    C = 1
    sigma = 0.1
    model = SVC(C=1, gamma=np.power(0.1, -2) / 2, kernel='rbf')
    model.fit(X, y.flatten())
    predictions = model.predict(Xval)
    b_accuracy = np.mean(np.double(predictions != yval))

    for c in range(1, 10, 1):
        for s in range(1, 11, 1):
            s /= 10
            model = SVC(C=c, gamma=np.power(s, -2) / 2, kernel='rbf')
            model.fit(X, y.flatten())
            predictions = model.predict(Xval)
            accuracy = np.mean(np.double(predictions != yval))
            if accuracy > b_accuracy:
                b_accuracy = accuracy
                C = c
                sigma = s

    return C, sigma
