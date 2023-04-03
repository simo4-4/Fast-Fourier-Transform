import DiscreteFourier as df
import time
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm


def test_1D_DFT():
    """
    Test the 1D DFT function
    """

    # test for X random arrays
    n = 50
    size = 1024
    for i in range(n):
        # Random array must have a power of 2 length
        data: np.ndarray = np.random.randint(0, 256, size)

        start = time.time()
        dft1 = df.DFT_1D_naive(data)
        print(f"Time for naive: {time.time() - start} seconds")

        dft2 = np.fft.fft(data)

        # compare the results
        equal = np.allclose(dft1, dft2)
        if not equal:
            print(
                f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {dft1.shape}, shape2: {dft2.shape}")
            break

        start = time.time()
        dft3 = df.FFT_1D(data, threshold=4)
        print(f"Time for FFT: {time.time() - start} seconds")

        equal = np.allclose(dft2, dft3)
        if not equal:
            print(
                f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {dft2.shape}, shape2: {dft3.shape}")
            break


def test_1D_IDFT():
    """
    Test the 1D IDFT function
    """

    # test for X random arrays
    n = 100
    size = 128
    for i in range(n):
        # Random array must have a power of 2 length
        data: np.ndarray = np.random.randint(0, 256, size)

        dft1 = df.IDFT_1D_naive(data)
        dft2 = np.fft.ifft(data)

        # compare the results
        equal = np.allclose(dft1, dft2)
        if not equal:
            print(
                f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {dft1.shape}, shape2: {dft2.shape}")
            break

        dft3 = df.IFFT_1D(data)
        equal = np.allclose(dft2, dft3)
        if not equal:
            print(
                f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {dft2.shape}, shape2: {dft3.shape}")
            break


def test_pad_signal():
    """
    Test the pad_signal function
    """

    for i in range(10, 100):
        # Random array must have a power of 2 length
        data: np.ndarray = np.random.randint(0, 256, (i, i))

        padded_signal = df.pad_signal(data)
        # print("Shape: ", data.shape, ", padded shape: ", padded_signal.shape)
        equal = np.allclose(data, padded_signal[:i, :i])

        r, c = padded_signal.shape
        correct_size = math.log2(r).is_integer()
        correct_size = correct_size and math.log2(c).is_integer() and r == c

        # Check is all extra values are 0
        correct_extra = np.allclose(padded_signal[i:, :], np.zeros((r - i, c)))
        correct_extra = correct_extra and np.allclose(padded_signal[:, i:], np.zeros((r, c - i)))
        correct_extra = correct_extra and np.allclose(padded_signal[i:, i:], np.zeros((r - i, c - i)))

        if not equal or not correct_size or not correct_extra:
            print(f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {padded_signal.shape}")
            break


def test_remove_padding():
    """
    Test the remove_padding function
    """

    for i in range(10, 100):
        # Random array must have a power of 2 length
        data: np.ndarray = np.random.randint(0, 256, (i, i))

        padded_signal = df.pad_signal(data)
        equal = np.allclose(data, df.remove_padding(data, padded_signal))

        if not equal:
            print(f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {padded_signal.shape}")
            break


def test_2D_DFT():
    """
    Test the 2D DFT function
    """

    n = 5
    size = (64, 64)
    for i in range(n):
        data: np.ndarray = np.random.randint(0, 256, size)

        dft2_np = np.fft.fft2(data)

        start = time.time()
        dft2 = df.FFT_2D(data, threshold=2)
        print(f"Time for FFT: {time.time() - start: 0.5f} seconds")

        start = time.time()
        df2n = df.DFT_2D_naive(data)
        print(f"Time for naive: {time.time() - start: 0.5f} seconds")

        # compare the results
        equal = np.allclose(dft2, dft2_np)
        if not equal:
            print(
                f"Arrays are not equal for array {i}, "
                f"shape: {data.shape}, shape1: {dft2.shape}, shape2: {dft2_np.shape}")
            print(dft2[:2, :2])
            print(dft2_np[:2, :2])
            break

        equal = np.allclose(dft2_np, df2n)
        if not equal:
            print(
                f"Arrays are not equal for array {i}, "
                f"shape: {data.shape}, shape1: {dft2_np.shape}, shape2: {df2n.shape}")
            break


def test_2D_IDFT():
    """
    Test the 2D DFT function
    """

    n = 100
    size = (128, 128)
    for i in range(n):
        data: np.ndarray = np.random.randint(0, 256, size)

        idft2 = df.IFFT_2D(data)
        idft2_np = np.fft.ifft2(data)

        # compare the results
        equal = np.allclose(idft2, idft2_np)
        if not equal:
            print(
                f"Arrays are not equal for array {i}, "
                f"shape: {data.shape}, shape1: {idft2.shape}, shape2: {idft2_np.shape}")
            break

def test_high_freq_filter():
    matrix = mpimg.imread("moonlanding.png")

    padding = df.pad_signal(matrix)
    fft = df.FFT_2D(padding)
    fft = np.fft.ifftshift(fft)

    denoise_levels = [0.405, 0.415, 0.42, 0.45, 0.48]

    # Show original image
    plt.figure(figsize=(8, 8))
    plt.title("Denoised Images")

    plt.subplot(3, 3, 1)
    plt.imshow(matrix, cmap='gray')
    plt.title("Original Image")

    for i, denoise_level in enumerate(denoise_levels):
        filter = df.inverse_rectangle_filter(fft.shape, denoise_level)
        fft_denoised = fft * filter
        fft_denoised = np.fft.fftshift(fft_denoised)
        fft_denoised = df.IFFT_2D(fft_denoised)
        fft_denoised = df.remove_padding(matrix, fft_denoised)

        plt.subplot(3, 3, i + 2)
        plt.imshow((np.abs(fft_denoised)), cmap='gray')

        # plt.imshow(np.log(np.abs(fft)), cmap='gray')
        plt.title(f"D Level: {denoise_level}")

    plt.show()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test the DFT functions')
    parser.add_argument('--all', action='store_true', help='Run the tests')
    parser.add_argument('-o', action='store_true', help='Run the 1D tests')
    parser.add_argument('-t', action='store_true', help='Run the 2D tests')
    parser.add_argument('-p', action='store_true', help='Run the padding tests')

    args = parser.parse_args()

    if args.all:
        print("Testing all functions")
        args.o = True
        args.t = True
        args.p = True

    if args.o:
        print("Testing 1D DFT")
        test_1D_DFT()

        print("Testing 1D IDFT")
        test_1D_IDFT()

    if args.p:
        print("Testing padding")
        test_pad_signal()

        print("Testing remove padding")
        test_remove_padding()

    if args.t:
        print("Testing 2D DFT")
        test_2D_DFT()

        print("Testing 2D IDFT")
        test_2D_IDFT()
