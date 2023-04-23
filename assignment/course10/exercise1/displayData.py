import numpy as np
import matplotlib.pyplot as plt


def display_data(X, example_width=None):
    # display_data Display 2D data in a nice grid
    #   [h, display_array] = display_data(X, example_width) displays 2D data
    #   stored in X in a nice grid. It returns the figure handle h and the
    #   displayed array if requested.

    # Set example_width automatically if not passed in
    if example_width is None:
        example_width = round(np.sqrt(X.shape[1]))

    # Gray Image
    plt.gray()

    # Compute rows, cols
    m, n = X.shape
    example_height = int(n / example_width)

    # Compute number of items to display
    display_rows = int(np.floor(np.sqrt(m)))
    display_cols = int(np.ceil(m / display_rows))

    # Between images padding
    pad = 1

    # Set up blank display
    display_array = - np.ones((pad + display_rows * (example_height + pad),
                               pad + display_cols * (example_width + pad)
                               ))

    curr_ex = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex >= m:
                break
            # Copy the patch

            # Get the max value of the patch
            max_val = np.max(np.abs(X[curr_ex, :]))
            display_array[pad + j * (example_height + pad) + np.arange(example_height)[:, None],
                          pad + i * (example_width + pad) + np.arange(example_width)] = \
                np.reshape(X[curr_ex, :], (example_height, example_width)) / max_val
            curr_ex += 1
        if curr_ex >= m:
            break

    h = plt.imshow(display_array, cmap='gray', vmin=-1, vmax=1)

    plt.axis('off')
    plt.show()
