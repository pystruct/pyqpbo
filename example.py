import matplotlib.pyplot as plt
import numpy as np
from pyqpbo import binary_grid, alpha_expansion_grid
from gco_python import cut_simple

from IPython.core.debugger import Tracer
tracer = Tracer()


def example_checkerboard():
    # generate a checkerboard
    x = np.ones((10, 12))
    x[::2, ::2] = -1
    x[1::2, 1::2] = -1
    x_noisy = x + np.random.normal(0, 1.5, size=x.shape)
    x_thresh = x_noisy > .0

    # create unaries
    unaries = x_noisy
    # as we convert to int, we need to multipy to get sensible values
    unaries = (10 * np.dstack([unaries, -unaries]).copy("C")).astype(np.int32)
    # create potts pairwise
    pairwise = 100 * np.eye(2, dtype=np.int32)

    # do simple cut
    result_qpbo = binary_grid(unaries, pairwise)
    # make non-labeled zero (instead of negative)
    result_qpbo_vis = result_qpbo.copy()
    result_qpbo_vis[result_qpbo < 0] = 0
    result_qpbo_vis[result_qpbo == 0] = -1

    # plot results
    plt.subplot(231, title="original")
    plt.imshow(x, interpolation='nearest')
    plt.subplot(232, title="noisy version")
    plt.imshow(x_noisy, interpolation='nearest')
    plt.subplot(233, title="rounded to integers")
    plt.imshow(unaries[:, :, 0], interpolation='nearest')
    plt.subplot(234, title="thresholding result")
    plt.imshow(x_thresh, interpolation='nearest')
    plt.subplot(235, title="qpbo")
    plt.imshow(result_qpbo_vis, interpolation='nearest')

    plt.show()


def example_binary():
    # generate trivial data
    x = np.ones((10, 12))
    x[:, 6:] = -1
    x_noisy = x + np.random.normal(0, 0.8, size=x.shape)
    x_thresh = x_noisy > .0

    # create unaries
    unaries = x_noisy
    # as we convert to int, we need to multipy to get sensible values
    unaries = (10 * np.dstack([unaries, -unaries]).copy("C")).astype(np.int32)
    # create potts pairwise
    pairwise = -10 * np.eye(2, dtype=np.int32)

    # do simple cut
    #result_qpbo = binary_grid(unaries, pairwise)
    result_qpbo = alpha_expansion_grid(unaries, pairwise)
    result_gc = cut_simple(unaries, pairwise)

    # plot results
    plt.subplot(231, title="original")
    plt.imshow(x, interpolation='nearest')
    plt.subplot(232, title="noisy version")
    plt.imshow(x_noisy, interpolation='nearest')
    plt.subplot(233, title="rounded to integers")
    plt.imshow(unaries[:, :, 0], interpolation='nearest')
    plt.subplot(234, title="thresholding result")
    plt.imshow(x_thresh, interpolation='nearest')
    plt.subplot(235, title="qpbo")
    plt.imshow(result_qpbo, interpolation='nearest')
    plt.subplot(236, title="graphcut")
    plt.imshow(result_gc, interpolation='nearest')
    plt.show()


def example_multinomial():
    # generate dataset with three stripes
    np.random.seed(15)
    x = np.zeros((10, 12, 3))
    x[:, :4, 0] = -1
    x[:, 4:8, 1] = -1
    x[:, 8:, 2] = -1
    unaries = x + 1.5 * np.random.normal(size=x.shape)
    x = np.argmin(x, axis=2)
    unaries = (unaries * 10).astype(np.int32)
    x_thresh = np.argmin(unaries, axis=2)

    # potts potential
    pairwise_potts = -2 * np.eye(3, dtype=np.int32)
    result = alpha_expansion_grid(unaries, 10 * pairwise_potts)
    # potential that penalizes 0-1 and 1-2 less thann 0-2
    pairwise_1d = -15 * np.eye(3, dtype=np.int32) - 8
    pairwise_1d[-1, 0] = 0
    pairwise_1d[0, -1] = 0
    print(pairwise_1d)
    result_1d = alpha_expansion_grid(unaries, pairwise_1d)
    plt.subplot(141, title="original")
    plt.imshow(x, interpolation="nearest")
    plt.subplot(142, title="thresholded unaries")
    plt.imshow(x_thresh, interpolation="nearest")
    plt.subplot(143, title="potts potentials")
    plt.imshow(result, interpolation="nearest")
    plt.subplot(144, title="1d topology potentials")
    plt.imshow(result_1d, interpolation="nearest")
    plt.show()


#example_binary()
#example_checkerboard()
example_multinomial()
#example_VH()
