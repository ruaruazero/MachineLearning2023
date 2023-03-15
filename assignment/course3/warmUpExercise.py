import numpy as np


def warm_up_exercise():
    """
    warm_up_exercise() is an example function that returns the 5x5 identity matrix
    :return: a
    """
    # ============= YOUR CODE HERE ==============
    # Instructions: Return the 5x5 identity matrix
    #               In octave, we return values by defining which variables
    #               represent the return values (at the top of the file)
    #               and then set them accordingly.
    a = np.identity(5)

    return a
    # ===========================================


if __name__ == '__main__':
    warm_up_exercise()
