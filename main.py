from matplotlib import cm

import DiscreteFourier as Df
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm


# function that takes the 2D ftt of an image filename and displays it
def mode_1(filename):
    original = mpimg.imread(filename)
    fft = Df.FFT_2D(original)

    # show original and fft
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap=cm.gray)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(np.abs(fft), cmap=cm.gray, norm=LogNorm(vmin=5))
    plt.title("2DFT of Image")

    plt.show()


# image is denoised by applying an FFT, truncating high frequencies and then displayed
def mode_2(filename):
    original = mpimg.imread(filename)
    fft = Df.FFT_2D(original)
    fft = Df.denoise(fft)
    inverse = Df.FFT_2D_inverse(fft)

    # show original and fft denoised
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap=cm.gray)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(inverse.real, cmap=cm.gray)
    plt.title("Denoised Image")

    plt.show()


if __name__ == '__main__':
    mode_1("moonlanding.png")
    #mode_2("moonlanding.png")
