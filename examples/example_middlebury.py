import Image
import matplotlib.pyplot as plt
import numpy as np
from pyqpbo import alpha_expansion_grid


def stereo_unaries(img1, img2):
    differences = []
    max_disp = 8
    for disp in np.arange(max_disp):
        if disp == 0:
            diff = np.sum((img1 - img2) ** 2, axis=2)
        else:
            diff = np.sum((img1[:, 2 * disp:, :] - img2[:, :-2 * disp, :]) **
                    2, axis=2)
        if disp != max_disp - 1:
            diff = diff[:, max_disp - disp - 1:disp - max_disp + 1]
        differences.append(diff)
    return np.dstack(differences).copy("C")


def potts_example():
    img1 = np.asarray(Image.open("scene1.row3.col1.ppm")) / 255.
    img2 = np.asarray(Image.open("scene1.row3.col2.ppm")) / 255.
    unaries = (stereo_unaries(img1, img2) * 100).astype(np.int32)
    n_disps = unaries.shape[2]

    #newshape = unaries.shape[:2]
    potts_cut = alpha_expansion_grid(unaries, -5 * np.eye(n_disps,
        dtype=np.int32), n_iter=5)
    x, y = np.ogrid[:n_disps, :n_disps]
    one_d_topology = np.abs(x - y).astype(np.int32).copy("C")
    one_d_cut = alpha_expansion_grid(unaries, 5 * one_d_topology, n_iter=5)
    plt.subplot(231, xticks=(), yticks=())
    plt.imshow(img1)
    plt.subplot(232, xticks=(), yticks=())
    plt.imshow(img1)
    plt.subplot(233, xticks=(), yticks=())
    plt.imshow(np.argmax(unaries, axis=2), interpolation='nearest')
    plt.subplot(223, xticks=(), yticks=())
    plt.imshow(potts_cut, interpolation='nearest')
    plt.subplot(224, xticks=(), yticks=())
    plt.imshow(one_d_cut, interpolation='nearest')
    plt.show()

potts_example()
