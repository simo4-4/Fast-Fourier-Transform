from matplotlib import cm

import DiscreteFourier as Df
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm

# function to compare values in an array and print values that are not equal to each other
def compare_arrays(array1, array2):
    for i in range(len(array1)):
        if math.floor(array1[i]) != math.floor(array2[i]):
            print(f"Index: {i} \t Array1: {array1[i]} \t Array2: {array2[i]}")


def test_1D_DFT(filename: str):
    """
    Test the 1D DFT function
    :param filename:
    :return:
    """

    # test for 100 random arrays
    for i in range(100):
        data: np.ndarray = np.random.randint(0, 100, 128)
        dft1 = Df.DFT_1D_naive(data)
        dft2 = np.fft.fft(data)

        # compare the results
        equal = np.allclose(dft1, dft2)
        if not equal:
            print(f"Arrays are not equal for array {i}")
            # Shape
            print(f"Shape 1: {dft1.shape}")
            print(f"Shape 2: {dft2.shape}")

            # Values
            print(f"Array 1: {dft1[:10]}")
            print(f"Array 2: {dft2[:10]}")
            break

        dft3 = Df.FFT_1D(data, threshold=16)
        equal = np.allclose(dft2, dft3)
        if not equal:
            print(f"Arrays are not equal for array {i}")
            # Shape
            print(f"Shape 2: {dft2.shape}")
            print(f"Shape 3: {dft3.shape}")

            # Values
            print(f"Array 2: {dft2[:10]}")
            print(f"Array 3: {dft3[:10]}")
            break



def test_mode_1(filename):
    # original = mpimg.imread(filename)
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
    # plt.subplot(3, 3, 1)
    # plt.imshow(original, cmap=cm.gray)
    # plt.title("Original")
    #
    # plt.subplot(3, 3, 2)
    # plt.imshow(fft, cmap=cm.gray, norm=LogNorm())
    # plt.title("2DFT us")
    #
    # plt.subplot(3, 3, 3)
    # plt.imshow(fft_np, cmap=cm.gray, norm=LogNorm())
    # plt.title("2DFT Numpy")

    # plt.subplot(3, 3, 4)
    # plt.imshow(inverse.real, cmap=cm.gray)
    # plt.title("Inverse of 2DFT of Image ECSE316")
    #
    # plt.subplot(3, 3, 5)
    # plt.imshow(inverse_np.real, cmap=cm.gray)
    # plt.title("Inverse of 2DFT using Numpy")
    #
    # plt.subplot(3, 3, 6)
    # plt.imshow(inverse_npm.real, cmap=cm.gray)
    # plt.title("Inverse of 2DFT using Numpy")
    #
    # plt.subplot(3, 3, 7)
    # plt.imshow(inverse_np2.real, cmap=cm.gray)
    # plt.title("Inverse of 2DFT using Numpy")

    plt.show()
