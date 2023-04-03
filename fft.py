from matplotlib import cm
import DiscreteFourier as df
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm
import argparse
import os

import testing


def mode_1(matrix: np.ndarray):
    """
    2D FFT of an image
    :param matrix: the image to be transformed
    """
    padding = df.pad_signal(matrix)
    fft = df.FFT_2D(padding)
    ifft = df.IFFT_2D(fft)
    ifft = ifft[:matrix.shape[0], :matrix.shape[1]]

    # show original and fft
    plt.subplot(2, 2, 1)
    plt.imshow(matrix, cmap='gray')
    plt.title("Original Image")

    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(padding), cmap='gray')
    plt.title("Padded Image")

    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(fft), cmap='gray', norm=LogNorm())
    plt.title("FFT")

    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(ifft), cmap='gray')
    plt.title("IFFT")

    if np.allclose(matrix, ifft):
        print("Images are equal")

    plt.show()


def mode_2(matrix: np.ndarray):
    padding = df.pad_signal(matrix)
    fft = df.FFT_2D(padding)
    fft = df.denoise(fft, 0.1)
    inverse = df.IFFT_2D(fft)
    inverse = df.remove_padding(matrix, inverse)

    plt.subplot(2, 2, 1)
    plt.imshow(matrix, cmap='gray')
    plt.title("Original Image")
    plt.subplot(2, 2, 2)
    plt.imshow(np.abs(padding), cmap='gray')
    plt.title("Padded Image")
    plt.subplot(2, 2, 3)
    plt.imshow(np.abs(fft), norm=LogNorm(vmin=5))
    plt.colorbar()
    plt.title("FFT of Image")
    plt.subplot(2, 2, 4)
    plt.imshow(np.abs(inverse), cmap='gray')
    plt.title("Denoised Image")

    plt.show()


def mode_3(matrix: np.ndarray):
    padding = df.pad_signal(matrix)
    fft = df.FFT_2D(padding)

    compression_levels = [0, 0.1, 0.2, 0.4, 0.65, 0.95]

    for i, compression_level in enumerate(compression_levels):
        fft_compressed = df.compress(fft, compression_level)
        inverse = df.IFFT_2D(fft_compressed)
        inverse = df.remove_padding(matrix, inverse)

        # Save fft_compressed non-zero coefficients to file
        np.savetxt(f"compressed_files/fft_compressed_{compression_level}.txt", fft_compressed[fft_compressed != 0])

        plt.subplot(2, 3, i + 1)
        plt.imshow(np.abs(inverse), cmap='gray')
        plt.title(f"C Level: {compression_level}")

    plt.show()


def mode_4():
    size = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
    ft_function = [df.DFT_2D_naive, df.FFT_2D]
    ft_name = ["2D DFT Naive", "2D FFT"]




def main():
    # Empty compressed_files directory using system command
    os.system("rm compressed_files/* > /dev/null 2>&1")

    testing.test_2D_DFT()



    # mode_1(mpimg.imread("moonlanding.png"))
    # mode_2(mpimg.imread("moonlanding.png"))
    # mode_3(mpimg.imread("moonlanding.png"))


if __name__ == '__main__':
    main()

