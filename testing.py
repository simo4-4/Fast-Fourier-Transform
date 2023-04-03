from random import random

from matplotlib import cm

import DiscreteFourier as Df
import time
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LogNorm


def test_1D_DFT():
    """
    Test the 1D DFT function
    """

    # test for X random arrays
    n = 5
    size = 1024
    for i in range(n):
        # Random array must have a power of 2 length
        data: np.ndarray = np.random.randint(0, 256, size)

        start = time.time()
        dft1 = Df.DFT_1D_naive(data)
        print(f"Time for naive: {time.time() - start} seconds")

        dft2 = np.fft.fft(data)

        # compare the results
        equal = np.allclose(dft1, dft2)
        if not equal:
            print(f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {dft1.shape}, shape2: {dft2.shape}")
            break

        start = time.time()
        dft3 = Df.FFT_1D(data, threshold=4)
        print(f"Time for FFT: {time.time() - start} seconds")

        equal = np.allclose(dft2, dft3)
        if not equal:
            print(f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {dft2.shape}, shape2: {dft3.shape}")
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

        dft1 = Df.IDFT_1D_naive(data)
        dft2 = np.fft.ifft(data)

        # compare the results
        equal = np.allclose(dft1, dft2)
        if not equal:
            print(f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {dft1.shape}, shape2: {dft2.shape}")
            break

        dft3 = Df.IFFT_1D(data)
        equal = np.allclose(dft2, dft3)
        if not equal:
            print(f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {dft2.shape}, shape2: {dft3.shape}")
            break


def test_pad_signal():
    """
    Test the pad_signal function
    """

    for i in range(10, 100):
        # Random array must have a power of 2 length
        data: np.ndarray = np.random.randint(0, 256, (i, i))

        padded_signal = Df.pad_signal(data)
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

        padded_signal = Df.pad_signal(data)
        equal = np.allclose(data, Df.remove_padding(data, padded_signal))

        if not equal:
            print(f"Arrays are not equal for array {i}, shape: {data.shape}, shape1: {padded_signal.shape}")
            break


def test_2D_DFT():
    """
    Test the 2D DFT function
    """

    n = 50
    size = (16, 16)
    for i in range(n):
        data: np.ndarray = np.random.randint(0, 256, size)

        dft2 = Df.FFT_2D(data)
        dft2_np = np.fft.fft2(data)
        df2n = Df.DFT_2D_naive(data)

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

        idft2 = Df.IFFT_2D(data)
        idft2_np = np.fft.ifft2(data)

        # compare the results
        equal = np.allclose(idft2, idft2_np)
        if not equal:
            print(
                f"Arrays are not equal for array {i}, "
                f"shape: {data.shape}, shape1: {idft2.shape}, shape2: {idft2_np.shape}")
            break
