from matplotlib import cm

import DiscreteFourier as Df
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# function to compare values in an array and print values that are not equal to each other
def compare_arrays(array1, array2):
    for i in range(len(array1)):
        if math.floor(array1[i]) != math.floor(array2[i]):
            print(f"Index: {i} \t Array1: {array1[i]} \t Array2: {array2[i]}")
def test_mode_1(filename):
    original = mpimg.imread(filename)
    original = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])

    fft = Df.FFT_2D_numpy(original)
    fft2 = Df.FFT_2D(original)
    fft_np = np.fft.fft2(original)

    # inverse using my inverse function
    inverse = Df.FFT_2D_inverse(fft)
    inverse_np = Df.FFT_2D_inverse(fft_np)

    # inverse using np inverse fucntion
    inverse_npm = np.fft.ifft2(fft)
    inverse_np2 = np.fft.ifft2(fft_np)

    # plot all the results
    plt.subplot(3, 3, 1)
    plt.imshow(original, cmap=cm.gray)
    plt.title("Original Image")

    plt.subplot(3, 3, 2)
    plt.imshow(np.log(np.abs(fft)), cmap=cm.gray)
    plt.title("2DFT of Image ECSE316")

    plt.subplot(3, 3, 3)
    plt.imshow(np.log(np.abs(fft_np)), cmap=cm.gray)
    plt.title("2DFT using Numpy")

    plt.subplot(3, 3, 4)
    plt.imshow(inverse.real, cmap=cm.gray)
    plt.title("Inverse of 2DFT of Image ECSE316")

    plt.subplot(3, 3, 5)
    plt.imshow(inverse_np.real, cmap=cm.gray)
    plt.title("Inverse of 2DFT using Numpy")

    plt.subplot(3, 3, 6)
    plt.imshow(inverse_npm.real, cmap=cm.gray)
    plt.title("Inverse of 2DFT using Numpy")

    plt.subplot(3, 3, 7)
    plt.imshow(inverse_np2.real, cmap=cm.gray)
    plt.title("Inverse of 2DFT using Numpy")

    plt.show()